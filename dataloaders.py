import os
import numpy as np
import torch


def build_sparse_adj_matrix(user_ids, item_ids, num_u, num_i):
    """Builds the normalized bipartite adjacency matrix D^{-0.5} A D^{-0.5} for LightGCN."""
    offset_item_ids = item_ids + num_u
    edge_src = torch.cat([user_ids, offset_item_ids])
    edge_dst = torch.cat([offset_item_ids, user_ids])
    values = torch.ones_like(edge_src, dtype=torch.float32)

    num_nodes = num_u + num_i
    degree = torch.zeros(num_nodes)
    degree.scatter_add_(0, edge_src, values)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    norm_values = d_inv_sqrt[edge_src] * values * d_inv_sqrt[edge_dst]
    indices = torch.stack([edge_src, edge_dst])
    return torch.sparse_coo_tensor(indices, norm_values, size=(num_nodes, num_nodes))


def maybe_to_csr(adj, num_nodes, dim, device):
    """Uses CSR sparse_mm if autograd is supported on this PyTorch/CUDA build."""
    try:
        adj_csr = adj.to_sparse_csr()
        probe = torch.randn(num_nodes, min(dim, 8), device=device, requires_grad=True)
        out = torch.sparse.mm(adj_csr, probe)
        out.sum().backward()
        return adj_csr
    except Exception:
        return adj


def _read_adj_txt(path):
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
    """Loads standard LightGCN/MACR format (train.txt, test.txt) from data_dir."""
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
        "mask_sets": train_sets,
        "test_sets": test_sets,
        "num_users": num_users,
        "num_items": num_items,
        "item_pop": torch.tensor(item_pop, dtype=torch.float32),
    }
