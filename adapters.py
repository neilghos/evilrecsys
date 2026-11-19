import torch
from dataloaders import build_sparse_adj_matrix


class BaseAdapter:
    """
    Base class for CF information-bottleneck adapters.
    """
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

    @property
    def num_u_side(self):
        return self.num_users

    @property
    def num_nodes(self):
        return self.num_u_side + self.num_items

    def _map_users(self, user_ids):
        raise NotImplementedError

    def process_batch(self, user_ids, item_ids, labels):
        u_indices = self._map_users(user_ids)
        i_indices = item_ids + self.num_u_side
        return u_indices, i_indices, labels

    def build_graph(self, user_ids, item_ids):
        u_indices = self._map_users(user_ids)
        return build_sparse_adj_matrix(u_indices, item_ids, self.num_u_side, self.num_items)


class UnfairnessAdapter(BaseAdapter):
    """Maps individual user_ids to protected group IDs."""
    def __init__(self, num_users, num_items, user_to_group_map):
        super().__init__(num_users, num_items)
        self.user_to_group = user_to_group_map
        self.num_groups = int(user_to_group_map.max().item()) + 1

    @property
    def num_u_side(self):
        return self.num_groups

    def _map_users(self, user_ids):
        return self.user_to_group[user_ids]


class PopularityAdapter(BaseAdapter):
    """Maps all users to a single global user node (index 0)."""
    @property
    def num_u_side(self):
        return 1

    def _map_users(self, user_ids):
        return torch.zeros_like(user_ids)


class ExposureAdapter(BaseAdapter):
    """Keeps user/item IDs intact and binarizes targets to observation O in {0, 1}."""
    def __init__(self, num_users, num_items, binarize_observation=True):
        super().__init__(num_users, num_items)
        self.binarize_observation = binarize_observation

    def _map_users(self, user_ids):
        return user_ids

    def process_batch(self, user_ids, item_ids, labels):
        u_indices, i_indices, out_labels = super().process_batch(user_ids, item_ids, labels)
        if self.binarize_observation:
            out_labels = (out_labels > 0).to(dtype=out_labels.dtype)
        return u_indices, i_indices, out_labels
