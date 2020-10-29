from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from torch.utils.data import Dataset
from numpy import load as np_load
import pickle
import torch
import numpy as np
import os
from torch import cat, zeros, ones, from_numpy, Tensor
from torch.utils.data.dataloader import DataLoader


class TagDataset(Dataset):

    def __init__(self, data_dir: Path, metadata_path: Path, split: AnyStr, class_num):
        super(TagDataset, self).__init__()
        the_dir: Path = data_dir.joinpath(split)

        self.examples: List[Path] = sorted(the_dir.iterdir())[::5]
        with open(metadata_path, 'rb') as f:
            self.tags = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        ex = np.load(str(ex), allow_pickle=True)
        features = ex['features'].item()
        tag = self.tags[item]
        return features, tag


def tag_collate_fn(batch):

    max_input_t_steps = max([i[0].shape[0] for i in batch])

    input_features = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]

    input_tensor = cat([
        cat([zeros(
            max_input_t_steps - i[0].shape[0],
            input_features).float(),
             from_numpy(i[0]).float()]).unsqueeze(0) for i in batch])

    output_tensor = cat([torch.Tensor(i[1]) for i in batch])

    return input_tensor, output_tensor


def tag_loader(data_dir, split, class_num, batch_size, shuffle=True):

    if split == 'development':
        metadata_path = Path('data/pickles/dev_keywords.p')
    elif split == 'evaluation':
        metadata_path = Path('data/pickles/eval_keywords.p')

    dataset = TagDataset(data_dir, metadata_path, split, class_num)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=tag_collate_fn
    )
