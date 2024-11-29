import torch


class ChoiceDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return feats, costs, sols, objs
