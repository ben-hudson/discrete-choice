from collections import namedtuple
from torch.utils.data import Dataset


class ChoiceDataset(Dataset):
    def __init__(self, feats, util, choices, choice_util, **kwargs):
        self.feats = feats
        self.util = util
        self.choices = choices
        self.choice_util = choice_util
        self.other = kwargs

        # this can't be pickled, so it only works with Dataloader.workers=0
        other_fields = list(kwargs.keys())
        self.datapoint_type = namedtuple(
            "ChoiceDatapoint",
            [
                "feats",
                "util",
                "choices",
                "choice_util",
            ]
            + other_fields,
        )

    def __getitem__(self, index):
        feats = self.feats[index]
        util = self.util[index]
        choices = self.choices[index]
        choice_util = self.choice_util[index]
        other = {key: value[index] for key, value in self.other.items()}
        return self.datapoint_type(feats, util, choices, choice_util, **other)

    def __len__(self):
        return len(self.util)
