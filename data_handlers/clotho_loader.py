#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, MutableMapping, Union, \
    Tuple, List, Callable, Optional
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
from torch import cat, zeros, from_numpy, ones, Tensor, LongTensor
from numpy import ndarray

from data_handlers._clotho import ClothoDataset
from tools.SpecAugment import spec_augment

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_clotho_loader']


def _clotho_collate_fn(batch: MutableSequence[ndarray], augment: bool) \
        -> Tuple[Tensor, Tensor, List[str]]:
    """Pads data.

    For each batch, the maximum input and output\
    time-steps are calculated. Then, then input and\
    output data are padded to match the maximum time-steps.

    The input data are padded with zeros in front, and\
    the output with] <EOS> tokens at the end.

    :param batch: Batch data of batch x time x features.\
                  First element in the list are the input\
                  data, second the output data.
    :type batch: list[numpy.ndarray]
    :return: Padded data. First tensor is the input data\
             and second the output.
    :rtype: torch.Tensor, torch.Tensor, list[str]
    """

    max_input_t_steps = 220  # max([i[0].shape[0] for i in batch])
    max_output_t_steps = max([i[1].shape[0] for i in batch])

    file_names = [i[2] for i in batch]

    input_features = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]
    PAD = 4367

    input_tensor = []
    for i in range(len(batch)):
        if batch[i][0].shape[0] > max_input_t_steps:
            t = from_numpy(batch[i][0][:max_input_t_steps]).unsqueeze(0)
        else:
            t = cat([from_numpy(batch[i][0]).float(),
                     zeros(max_input_t_steps - batch[i][0].shape[0], input_features).float()
                     ]).unsqueeze(0)
        input_tensor.append(t)
    input_tensor = cat(input_tensor)

    if augment:
        input_tensor = spec_augment(input_tensor)

    output_tensor = cat([
        cat([
            from_numpy(i[1]).long(),
            ones(max_output_t_steps - len(i[1])).mul(PAD).long()
        ]).unsqueeze(0) for i in batch])
    *_, output_len = zip(*batch)
    output_len = LongTensor(output_len)

    return input_tensor, output_tensor, output_len, file_names


def get_clotho_loader(split: str,
                      is_training: bool,
                      data_dir: str,
                      input_field_name: str,
                      output_field_name: str,
                      batch_size: int,
                      num_workers: Optional[int] = 1,
                      load_into_memory: bool = True,
                      shuffle: Optional[bool] = True,
                      drop_last: Optional[bool] = True,
                      augment: Optional[bool] = False) \
        -> DataLoader:
    """Gets the data loader.

    :param augment:
    :param split: Split to be used.
    :type split: str
    :param is_training: Is training data?
    :type is_training: bool
    :param settings_data: Data loading and dataset settings.
    :type settings_data: dict
    :param settings_io: Files I/O settings.
    :type settings_io: dict
    :return: Data loader.
    :rtype: torch.utils.data.DataLoader
    """

    dataset = ClothoDataset(
        data_dir=data_dir,
        split=split,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
        load_into_memory=load_into_memory)

    shuffle = shuffle if is_training else False
    drop_last = drop_last if is_training else False

    collate_fn: Callable = partial(
        _clotho_collate_fn,
        augment=augment
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn)

# EOF
