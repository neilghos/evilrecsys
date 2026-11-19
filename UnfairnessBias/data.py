import os
import numpy as np
import torch


def _load_cpfair_dir(data_dir):
    name = os.path.basename(os.path.normpath(data_dir))
    train_path = os.path.join(data_dir, f"{name}_train.txt")
    test_path = os.path.join(data_dir, f"{name}_test.txt")
    active_path = os.path.join(data_dir, "groups", "users", "020", "active_ids.txt")

    train_sets, test_sets = {}, {}
    max_u, max_i = 0, 0
    for path, target in [(train_path, train_sets), (test_path, test_sets)]:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u, i = int(parts[0]), int(parts[1])
                target.setdefault(u, set()).add(i)
                if u > max_u:
                    max_u = u
                if i > max_i:
                    max_i = i

    num_users = max_u + 1
    num_items = max_i + 1
    groups = np.zeros(num_users, dtype=np.int64)
    with open(active_path, "r") as f:
        for tok in f.read().split():
            uid = int(tok)
            if 0 <= uid < num_users:
                groups[uid] = 1
    return train_sets, test_sets, groups, num_users, num_items


def load_unfairness_dataset(data_dir):
    """
    Loads UnfairnessBias benchmarks in either:
      - FDA format (`ml`): train.npy, test.npy, users_features.npy (col 0 = gender)
      - CPFairRecSys format (`BookCrossing`, `AmazonOffice`, `LastFM`):
        <name>_train.txt, <name>_test.txt, groups/users/020/active_ids.txt (1=active, 0=inactive)
    """
    if os.path.exists(os.path.join(data_dir, "train.npy")):
        train_dict, _ = np.load(os.path.join(data_dir, "train.npy"), allow_pickle=True)
        test_dict, _ = np.load(os.path.join(data_dir, "test.npy"), allow_pickle=True)
        users_features = np.load(os.path.join(data_dir, "users_features.npy"))
        groups = np.asarray(users_features[:, 0], dtype=np.int64)
        num_users = len(groups)
        max_item = 0
        train_sets, test_sets = {}, {}
        for u, items in train_dict.items():
            s = set(int(i) for i in items)
            if s:
                train_sets[int(u)] = s
                max_item = max(max_item, max(s))
        for u, items in test_dict.items():
            s = set(int(i) for i in items)
            if s:
                test_sets[int(u)] = s
                max_item = max(max_item, max(s))
        num_items = max_item + 1
    else:
        train_sets, test_sets, groups, num_users, num_items = _load_cpfair_dir(data_dir)

    train_u, train_i = [], []
    for u in sorted(train_sets.keys()):
        for i in sorted(train_sets[u]):
            train_u.append(u)
            train_i.append(i)

    item_pop = np.zeros(num_items, dtype=np.float32)
    for i in train_i:
        item_pop[i] += 1.0

    # Group-collapsed interaction sets for the Demographic/Consumer-Group Evil Teacher
    num_groups = int(groups.max()) + 1
    group_sets = {g: set() for g in range(num_groups)}
    group_u, group_i = [], []
    for u, i in zip(train_u, train_i):
        g = int(groups[u])
        group_u.append(g)
        group_i.append(i)
        group_sets[g].add(i)

    return {
        "train_users": torch.tensor(train_u, dtype=torch.long),
        "train_items": torch.tensor(train_i, dtype=torch.long),
        "train_sets": train_sets,
        "test_sets": test_sets,
        "user_groups": torch.tensor(groups, dtype=torch.long),
        "num_groups": num_groups,
        "group_users": torch.tensor(group_u, dtype=torch.long),
        "group_items": torch.tensor(group_i, dtype=torch.long),
        "group_sets": group_sets,
        "num_users": num_users,
        "num_items": num_items,
        "item_pop": torch.tensor(item_pop, dtype=torch.float32),
    }
