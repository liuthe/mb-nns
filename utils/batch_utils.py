#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py: Utility functions for the project.
"""

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (Tensor of (num_sample, 2*L*L )): Tensor of size (num_sample, 2*L*L) containing the source and target sentences
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    num_samples = data.size(0)
    batch_num = math.ceil(num_samples / batch_size)
    index_array = list(range(num_samples))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        yield data[indices]