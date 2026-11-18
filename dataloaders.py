import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RecSysDataset(Dataset):
    """
    A generic PyTorch dataset for Recommender Systems.
    It yields raw (user, item, label) interactions. The adapters will 
    handle transforming these into the restricted inputs for the Evil Teacher.
    """
    def __init__(self, user_ids, item_ids, labels, num_items, num_negative_samples=1):
        """
        Args:
            user_ids (np.array or torch.Tensor): Array of user indices.
            item_ids (np.array or torch.Tensor): Array of item indices.
            labels (np.array or torch.Tensor): Array of interaction labels (e.g., 1 for implicit).
            num_items (int): Total number of items in the dataset (used for negative sampling).
            num_negative_samples (int): How many negative samples per positive interaction.
        """
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        self.num_items = num_items
        self.num_negative_samples = num_negative_samples
        
        # If we need on-the-fly negative sampling, we do it in __getitem__
        # For simplicity in this skeleton, we assume the dataset already contains
        # the positive interactions, and we sample negatives on the fly.

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u = self.user_ids[idx]
        i = self.item_ids[idx]
        label = self.labels[idx]
        
        # Note: In a real training loop, you might want to return 1 positive 
        # and N negative items per user for BPR loss. 
        # For a standard Cross-Entropy / MSE loss, returning (u, i, label) is fine.
        return u, i, label

def get_dataloader(user_ids, item_ids, labels, num_items, batch_size=1024, shuffle=True):
    """
    Helper function to get a PyTorch DataLoader.
    """
    dataset = RecSysDataset(user_ids, item_ids, labels, num_items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example of how to construct the adjacency matrix for LightGCN
def build_sparse_adj_matrix(user_ids, item_ids, num_u, num_i):
    """
    Builds the normalized bipartite adjacency matrix for LightGCN.
    Expected matrix shape: (num_u + num_i) x (num_u + num_i)
    """
    # Offset items by num_users
    offset_item_ids = item_ids + num_u
    
    # Edges (user -> item and item -> user)
    edge_src = torch.cat([user_ids, offset_item_ids])
    edge_dst = torch.cat([offset_item_ids, user_ids])
    
    # Values (1.0 for each interaction)
    values = torch.ones_like(edge_src, dtype=torch.float32)
    
    num_nodes = num_u + num_i
    
    # Calculate degree for normalization: D^{-0.5} * A * D^{-0.5}
    degree = torch.zeros(num_nodes)
    degree.scatter_add_(0, edge_src, values)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    norm_values = d_inv_sqrt[edge_src] * values * d_inv_sqrt[edge_dst]
    
    indices = torch.stack([edge_src, edge_dst])
    adj_matrix = torch.sparse_coo_tensor(indices, norm_values, size=(num_nodes, num_nodes))
    
    return adj_matrix


class NewsRecDataset(Dataset):
    """
    PyTorch Dataset for impression-logged News Recommendation (e.g., MIND, Adressa).
    Yields a dict per impression so both standard NRMS and News Adapters can
    selectively mask/bottleneck user history, text, category, age, or position.
    """
    def __init__(
        self,
        history_tokens,
        history_mask,
        candidate_tokens,
        labels,
        history_cats=None,
        candidate_cats=None,
        candidate_ages=None,
        candidate_positions=None,
        user_group=None,
        use_plm=False,
    ):
        token_dtype = torch.float32 if use_plm else torch.long
        self.history_tokens = torch.as_tensor(history_tokens, dtype=token_dtype)
        self.history_mask = torch.as_tensor(history_mask, dtype=torch.bool)
        self.candidate_tokens = torch.as_tensor(candidate_tokens, dtype=token_dtype)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

        num_samples, hist_len = self.history_tokens.shape[:2]
        num_cands = self.candidate_tokens.shape[1]

        self.history_cats = (
            torch.as_tensor(history_cats, dtype=torch.long)
            if history_cats is not None
            else torch.zeros((num_samples, hist_len), dtype=torch.long)
        )
        self.candidate_cats = (
            torch.as_tensor(candidate_cats, dtype=torch.long)
            if candidate_cats is not None
            else torch.zeros((num_samples, num_cands), dtype=torch.long)
        )
        self.candidate_ages = (
            torch.as_tensor(candidate_ages, dtype=torch.long)
            if candidate_ages is not None
            else torch.zeros((num_samples, num_cands), dtype=torch.long)
        )
        self.candidate_positions = (
            torch.as_tensor(candidate_positions, dtype=torch.long)
            if candidate_positions is not None
            else torch.arange(num_cands, dtype=torch.long).unsqueeze(0).expand(num_samples, -1)
        )
        self.user_group = (
            torch.as_tensor(user_group, dtype=torch.long)
            if user_group is not None
            else torch.zeros(num_samples, dtype=torch.long)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "history_tokens": self.history_tokens[idx],
            "history_mask": self.history_mask[idx],
            "history_cats": self.history_cats[idx],
            "candidate_tokens": self.candidate_tokens[idx],
            "candidate_cats": self.candidate_cats[idx],
            "candidate_ages": self.candidate_ages[idx],
            "candidate_positions": self.candidate_positions[idx],
            "user_group": self.user_group[idx],
            "labels": self.labels[idx],
        }


def get_news_dataloader(dataset, batch_size=64, shuffle=True):
    """Helper function to get a PyTorch DataLoader for NewsRecDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _read_adj_txt(path):
    """Reads space-separated `user_id item1 item2 ...` lines into {u: [items]}."""
    interactions = {}
    max_u, max_i = 0, 0
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            interactions[u] = items
            max_u = max(max_u, u)
            max_i = max(max_i, max(items))
    return interactions, max_u, max_i


def load_cf_txt_dataset(data_dir):
    """
    Loads standard LightGCN/MACR format (train.txt, test.txt) from data_dir.
    Returns dict with train tensors, user->item sets, num_users, num_items, item_pop.
    """
    import os
    train_dict, tr_u, tr_i = _read_adj_txt(os.path.join(data_dir, "train.txt"))
    test_dict, te_u, te_i = _read_adj_txt(os.path.join(data_dir, "test.txt"))

    num_users = max(tr_u, te_u) + 1
    num_items = max(tr_i, te_i) + 1

    u_list, i_list = [], []
    item_pop = np.zeros(num_items, dtype=np.int64)
    train_sets = {}
    for u, items in train_dict.items():
        train_sets[u] = set(items)
        for i in items:
            u_list.append(u)
            i_list.append(i)
            item_pop[i] += 1

    test_sets = {u: set(items) for u, items in test_dict.items()}
    return {
        "train_users": torch.tensor(u_list, dtype=torch.long),
        "train_items": torch.tensor(i_list, dtype=torch.long),
        "train_sets": train_sets,
        "test_sets": test_sets,
        "num_users": num_users,
        "num_items": num_items,
        "item_pop": torch.tensor(item_pop, dtype=torch.float32),
    }


class BPRDataset(Dataset):
    """Samples (user, pos_item, neg_item) triplets for BPR training."""
    def __init__(self, train_users, train_items, train_sets, num_items):
        self.train_users = train_users
        self.train_items = train_items
        self.train_sets = train_sets
        self.num_items = num_items

    def __len__(self):
        return len(self.train_users)

    def __getitem__(self, idx):
        u = int(self.train_users[idx].item())
        pos_i = int(self.train_items[idx].item())
        pos_set = self.train_sets[u]
        while True:
            neg_i = np.random.randint(0, self.num_items)
            if neg_i not in pos_set:
                break
        return u, pos_i, neg_i



