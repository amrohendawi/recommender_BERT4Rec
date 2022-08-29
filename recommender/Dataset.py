import torch
import random
from recommender.data_processing import get_context, pad_list, MASK


def mask_list(l1, p=0.8):
    l1 = [a if random.random() < p else MASK for a in l1]

    return l1


def mask_last_elements_list(l1, val_context_size: int = 5):
    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)

    return l1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, history_size=120):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.history_size = history_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        context = get_context(df, split=self.split, context_size=self.history_size)

        trg_items = context["movieId_mapped"].tolist()

        if self.split == "train":
            src_items = mask_list(trg_items)
        else:
            src_items = mask_last_elements_list(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
        src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)

        src_items = torch.tensor(src_items, dtype=torch.long)

        trg_items = torch.tensor(trg_items, dtype=torch.long)

        return src_items, trg_items
