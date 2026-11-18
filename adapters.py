import torch
from dataloaders import build_sparse_adj_matrix


class BaseAdapter:
    """
    Base class for adapters. Adapters act as information bottlenecks,
    modifying the data batch before it reaches the Evil Teacher.
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

    def process_batch(self, user_ids, item_ids, labels, user_features=None, item_features=None):
        """
        Takes raw batch and returns modified (u_indices, i_indices_offset, labels)
        aligned with the bipartite graph node space.
        """
        u_indices = self._map_users(user_ids)
        i_indices = item_ids + self.num_u_side
        return u_indices, i_indices, labels

    def build_graph(self, user_ids, item_ids):
        """
        Builds the restricted bipartite graph for this specific bias type.
        Returns a sparse adjacency matrix of shape (num_nodes, num_nodes).
        """
        u_indices = self._map_users(user_ids)
        return build_sparse_adj_matrix(u_indices, item_ids, self.num_u_side, self.num_items)


class UnfairnessAdapter(BaseAdapter):
    """
    Forces the model to learn pure group/demographic unfairness stereotypes.
    Replaces individual user_ids with their protected group ID.
    """
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
    """
    Forces the model to learn pure item popularity.
    Completely blinds the model to the user by mapping ALL users to a single
    'global_user' node (index 0).
    """
    @property
    def num_u_side(self):
        return 1

    def _map_users(self, user_ids):
        return torch.zeros_like(user_ids)


class ExposureAdapter(BaseAdapter):
    """
    Forces the model to learn exposure propensity.
    Keeps user and item IDs intact, but changes the learning objective
    to predict exposure (1 if observed, 0 if negative sample).
    """
    def __init__(self, num_users, num_items, binarize_observation=True):
        super().__init__(num_users, num_items)
        self.binarize_observation = binarize_observation

    def _map_users(self, user_ids):
        return user_ids

    def process_batch(self, user_ids, item_ids, labels, user_features=None, item_features=None):
        u_indices, i_indices, out_labels = super().process_batch(
            user_ids, item_ids, labels, user_features, item_features
        )
        if self.binarize_observation:
            out_labels = (out_labels > 0).to(dtype=out_labels.dtype)
        return u_indices, i_indices, out_labels


# ==============================================================================
# Front 2: News Recommendation Adapters (Feature / Channel Bottlenecks)
# ==============================================================================

class BaseNewsAdapter:
    """
    Information-bottleneck adapter for impression-logged News Rec (e.g., MIND).
    Controls which input channels EvilNRMS is allowed to see.
    """
    use_text = True
    use_history = True
    use_cats = False
    use_ages = False
    use_positions = False
    use_group = False

    def process_batch(self, batch):
        """Returns a bottlenecked copy of the batch dict with active channel flags."""
        return {
            "history_tokens": batch["history_tokens"] if (self.use_history and self.use_text) else torch.zeros_like(batch["history_tokens"]),
            "history_mask": batch["history_mask"] if self.use_history else torch.zeros_like(batch["history_mask"]),
            "history_cats": batch["history_cats"] if (self.use_history and self.use_cats) else torch.zeros_like(batch["history_cats"]),
            "candidate_tokens": batch["candidate_tokens"] if self.use_text else torch.zeros_like(batch["candidate_tokens"]),
            "candidate_cats": batch["candidate_cats"] if self.use_cats else torch.zeros_like(batch["candidate_cats"]),
            "candidate_ages": batch["candidate_ages"] if self.use_ages else torch.zeros_like(batch["candidate_ages"]),
            "candidate_positions": batch["candidate_positions"] if self.use_positions else torch.zeros_like(batch["candidate_positions"]),
            "user_group": batch["user_group"] if self.use_group else torch.zeros_like(batch["user_group"]),
            "labels": batch["labels"],
            "_channels": {
                "use_text": self.use_text,
                "use_history": self.use_history,
                "use_cats": self.use_cats,
                "use_ages": self.use_ages,
                "use_positions": self.use_positions,
                "use_group": self.use_group,
            },
        }


class PassthroughNewsAdapter(BaseNewsAdapter):
    """Unrestricted news adapter for a standard NRMS teacher or student."""
    use_text = True
    use_history = True
    use_cats = True


class NewsPopularityAdapter(BaseNewsAdapter):
    """Blinds user history so the model only learns item-level popularity/clickability."""
    use_text = True
    use_history = False


class NewsUnfairnessAdapter(BaseNewsAdapter):
    """Replaces user click history with protected group ID."""
    use_text = True
    use_history = False
    use_group = True


class NewsTopicAdapter(BaseNewsAdapter):
    """Masks out article text so the model only learns user-category / filter-bubble bias."""
    use_text = False
    use_history = True
    use_cats = True


class NewsFreshnessAdapter(BaseNewsAdapter):
    """Blinds user and text so the model only learns recency / publish-age bias."""
    use_text = False
    use_history = False
    use_ages = True


class NewsPositionAdapter(BaseNewsAdapter):
    """Blinds user and text so the model only learns impression rank / position bias."""
    use_text = False
    use_history = False
    use_positions = True


