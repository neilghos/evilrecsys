import torch
import torch.nn as nn
import torch.nn.functional as F

class EvilLightGCN(nn.Module):
    """
    A generic LightGCN model that acts as the 'Evil Teacher'.
    It is bias-agnostic. It simply takes an adjacency matrix and initial 
    embeddings (which are filtered/provided by the Adapters) and performs 
    graph convolutions.
    """
    def __init__(self, num_nodes, embedding_dim, num_layers=3):
        super(EvilLightGCN, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Base embeddings for all nodes (users, items, demographics, etc.)
        # The adapter decides which indices correspond to which entities.
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
    def forward(self, adj_matrix):
        """
        Performs LightGCN message passing.
        
        Args:
            adj_matrix (torch.sparse.FloatTensor): The normalized adjacency matrix.
            
        Returns:
            torch.Tensor: The final representations (average of all layers).
        """
        # initial embeddings E^0
        all_emb = self.embedding.weight
        embs = [all_emb]
        
        # Message passing layers
        for _ in range(self.num_layers):
            # E^{k+1} = A * E^k
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
            
        # Combine representations (LightGCN averages layer outputs)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        return final_emb

    def predict(self, u_indices, i_indices, adj_matrix):
        """
        Predicts scores for given pairs.
        
        Args:
            u_indices (torch.Tensor): Indices of the 'user' side (could be demographic group ID).
            i_indices (torch.Tensor): Indices of the 'item' side.
            adj_matrix (torch.sparse.FloatTensor): The normalized adjacency matrix.
            
        Returns:
            torch.Tensor: Predicted scores.
        """
        final_emb = self.forward(adj_matrix)
        
        u_emb = final_emb[u_indices]
        i_emb = final_emb[i_indices]
        
        # Inner product for prediction
        scores = torch.sum(u_emb * i_emb, dim=1)
        return scores


# ==============================================================================
# Front 2: News Recommendation Evil Teacher (NRMS + Adapter Channel Bottlenecks)
# ==============================================================================

class AdditiveAttention(nn.Module):
    """Additive attention pooling: collapses [B, Seq, D] -> [B, D]."""
    def __init__(self, input_dim, query_dim=200):
        super().__init__()
        self.proj = nn.Linear(input_dim, query_dim)
        self.query = nn.Linear(query_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: [B, S, D], mask: [B, S] bool (True = valid, False = pad)
        attn_logits = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # [B, S]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
        weights = F.softmax(attn_logits, dim=-1)  # [B, S]
        if mask is not None:
            # Guard all-padded histories so softmax doesn't leak uniform weights
            has_valid = mask.any(dim=-1, keepdim=True)
            weights = weights * has_valid.to(weights.dtype)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class NewsEncoder(nn.Module):
    """
    Encodes a batch of news items into [B, embed_dim].
    Supports both classic word-ID NRMS (GloVe/WordEmb + MultiHeadSelfAttention)
    and frozen PLM embeddings (plm_dim -> Linear + LayerNorm), plus optional
    category embeddings for TopicAdapter.
    """
    def __init__(
        self,
        embed_dim=256,
        num_heads=16,
        query_dim=200,
        vocab_size=None,
        word_embed_dim=300,
        pretrained_word_emb=None,
        plm_dim=None,
        num_categories=0,
        dropout=0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_plm = plm_dim is not None

        if self.use_plm:
            self.plm_proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(plm_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            assert vocab_size is not None, "Provide vocab_size when plm_dim is None."
            self.word_emb = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
            if pretrained_word_emb is not None:
                self.word_emb.weight.data.copy_(pretrained_word_emb)
            self.word_proj = (
                nn.Linear(word_embed_dim, embed_dim)
                if word_embed_dim != embed_dim
                else nn.Identity()
            )
            self.word_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.word_pool = AdditiveAttention(embed_dim, query_dim)
            self.dropout = nn.Dropout(dropout)

        self.cat_emb = nn.Embedding(num_categories, embed_dim) if num_categories > 0 else None

    def forward(self, tokens, cats=None, use_text=True, use_cats=False):
        out = None
        if use_text:
            if self.use_plm:
                out = self.plm_proj(tokens)
            else:
                pad_mask = tokens != 0
                x = self.dropout(self.word_proj(self.word_emb(tokens)))
                attn_out, _ = self.word_attn(x, x, x, key_padding_mask=~pad_mask)
                out = self.word_pool(self.dropout(attn_out), mask=pad_mask)

        if use_cats and self.cat_emb is not None and cats is not None:
            cat_vec = self.cat_emb(cats)
            out = cat_vec if out is None else (out + cat_vec)

        if out is None:
            out = torch.zeros(tokens.size(0), self.embed_dim, device=tokens.device)
        return out


class UserEncoder(nn.Module):
    """Encodes a user's clicked news sequence [B, K, D] -> [B, D]."""
    def __init__(self, embed_dim=256, num_heads=16, query_dim=200, dropout=0.2):
        super().__init__()
        self.hist_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.hist_pool = AdditiveAttention(embed_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hist_emb, hist_mask):
        # MultiheadAttention expects key_padding_mask=True for padded positions
        attn_out, _ = self.hist_attn(hist_emb, hist_emb, hist_emb, key_padding_mask=~hist_mask)
        return self.hist_pool(self.dropout(attn_out), mask=hist_mask)


class EvilNRMS(nn.Module):
    """
    Bias-agnostic NRMS backbone for News Recommendation.
    Reads channel flags injected by BaseNewsAdapter so the same model acts as
    a normal NRMS or an Evil Teacher for Popularity, Unfairness, Topic,
    Freshness, or Position bias.
    """
    def __init__(
        self,
        embed_dim=256,
        num_heads=16,
        query_dim=200,
        vocab_size=None,
        word_embed_dim=300,
        pretrained_word_emb=None,
        plm_dim=None,
        num_categories=0,
        num_groups=1,
        num_age_buckets=0,
        max_positions=0,
        dropout=0.2,
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_dim=query_dim,
            vocab_size=vocab_size,
            word_embed_dim=word_embed_dim,
            pretrained_word_emb=pretrained_word_emb,
            plm_dim=plm_dim,
            num_categories=num_categories,
            dropout=dropout,
        )
        self.user_encoder = UserEncoder(embed_dim, num_heads, query_dim, dropout)
        self.global_user_emb = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.group_emb = nn.Embedding(num_groups, embed_dim)
        self.age_bias = nn.Embedding(num_age_buckets, 1) if num_age_buckets > 0 else None
        self.pos_bias = nn.Embedding(max_positions, 1) if max_positions > 0 else None

    def forward(self, batch):
        ch = batch.get("_channels", {})
        use_text = ch.get("use_text", True)
        use_history = ch.get("use_history", True)
        use_cats = ch.get("use_cats", False)
        use_ages = ch.get("use_ages", False)
        use_positions = ch.get("use_positions", False)
        use_group = ch.get("use_group", False)

        cand_tokens = batch["candidate_tokens"]
        bsz, num_cands = cand_tokens.shape[:2]

        # 1. Encode candidates: [B, C, D]
        flat_cand_tok = cand_tokens.reshape(bsz * num_cands, *cand_tokens.shape[2:])
        flat_cand_cat = batch["candidate_cats"].reshape(bsz * num_cands)
        cand_vecs = self.news_encoder(
            flat_cand_tok, flat_cand_cat, use_text=use_text, use_cats=use_cats
        ).view(bsz, num_cands, -1)

        # 2. Encode user side: [B, D]
        if use_history:
            hist_tokens = batch["history_tokens"]
            hist_len = hist_tokens.size(1)
            flat_hist_tok = hist_tokens.reshape(bsz * hist_len, *hist_tokens.shape[2:])
            flat_hist_cat = batch["history_cats"].reshape(bsz * hist_len)
            hist_vecs = self.news_encoder(
                flat_hist_tok, flat_hist_cat, use_text=use_text, use_cats=use_cats
            ).view(bsz, hist_len, -1)
            user_vec = self.user_encoder(hist_vecs, batch["history_mask"])
        elif use_group:
            user_vec = self.group_emb(batch["user_group"])
        else:
            user_vec = self.global_user_emb.expand(bsz, -1)

        # 3. Dot-product scores + optional context/bias heads: [B, C]
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        if use_ages and self.age_bias is not None:
            scores = scores + self.age_bias(batch["candidate_ages"]).squeeze(-1)
        if use_positions and self.pos_bias is not None:
            scores = scores + self.pos_bias(batch["candidate_positions"]).squeeze(-1)
        return scores

