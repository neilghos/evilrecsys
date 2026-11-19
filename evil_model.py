import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """Sets Python, NumPy, and PyTorch (CPU/CUDA) RNGs for end-to-end determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class LightGCN(nn.Module):
    """
    End-to-end LightGCN backbone shared across all bias folders and baselines
    (PopBias, SelectionBias, ExposureBias, UnfairnessBias).
    """
    def __init__(self, num_nodes, embedding_dim, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        all_emb = self.embedding.weight
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        return torch.mean(torch.stack(embs, dim=1), dim=1)

    def predict(self, u_indices, i_indices, adj_matrix):
        final_emb = self.forward(adj_matrix)
        return torch.sum(final_emb[u_indices] * final_emb[i_indices], dim=1)


class FastLightGCNSampler:
    """
    GPU-vectorized equivalent of LightGCN-PyTorch's UniformSample_original.
    When item_weights is None: picks positive items uniformly per user.
    When item_weights is provided (shape [num_items]): picks positives proportional
    to item_weights[i] per user.
    """
    def __init__(self, train_sets, num_users, num_items, device, item_weights=None):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.use_weights = item_weights is not None

        valid_users = [u for u in range(num_users) if len(train_sets.get(u, ())) > 0]
        self.valid_users = torch.tensor(valid_users, dtype=torch.long, device=device)

        w_np = item_weights.cpu().numpy().astype(np.float64) if self.use_weights else None
        offsets = np.zeros(num_users, dtype=np.int64)
        lens = np.zeros(num_users, dtype=np.int64)
        flat_items = []
        flat_cdfs = []

        cursor = 0
        for u in range(num_users):
            items = sorted(train_sets.get(u, ()))
            offsets[u] = cursor
            lens[u] = len(items)
            if len(items) > 0:
                flat_items.extend(items)
                if self.use_weights:
                    w = np.maximum(w_np[items], 1e-8)
                    cdf = np.cumsum(w)
                    flat_cdfs.extend((cdf / cdf[-1]).tolist())
                cursor += len(items)

        self.train_size = cursor
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.lens = torch.tensor(lens, dtype=torch.long, device=device)
        self.flat_items = torch.tensor(flat_items, dtype=torch.long, device=device)

        if self.use_weights:
            flat_cdfs_t = torch.tensor(flat_cdfs, dtype=torch.float32, device=device)
            max_deg = int(lens.max())
            cdf_mat = torch.ones((num_users, max_deg), dtype=torch.float32, device=device)
            item_mat = torch.zeros((num_users, max_deg), dtype=torch.long, device=device)
            for u in valid_users:
                l = int(lens[u])
                s = int(offsets[u])
                cdf_mat[u, :l] = flat_cdfs_t[s : s + l]
                item_mat[u, :l] = self.flat_items[s : s + l]
            self.cdf_mat = cdf_mat
            self.item_mat = item_mat

    def sample(self):
        u_idx = torch.randint(0, len(self.valid_users), (self.train_size,), device=self.device)
        users = self.valid_users[u_idx]
        rand_u = torch.rand(self.train_size, device=self.device)

        if not self.use_weights:
            pos_offsets = (rand_u * self.lens[users].float()).long().clamp(max=self.lens[users] - 1)
            pos_items = self.flat_items[self.offsets[users] + pos_offsets]
        else:
            cdfs = self.cdf_mat[users]
            col_idx = torch.searchsorted(cdfs, rand_u.unsqueeze(1)).squeeze(1)
            col_idx = col_idx.clamp(max=self.lens[users] - 1)
            pos_items = self.item_mat[users, col_idx]

        neg_items = torch.randint(0, self.num_items, (self.train_size,), device=self.device)
        return users, pos_items, neg_items
