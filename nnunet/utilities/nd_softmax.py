#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)


def numpy_softmax(x, axis=-1):
    """
    softmax numpy version
    :param x: input shape [B,D,H,W,C]
    :return: [B,D,H,W,C]
    """
    x = x - np.expand_dims(x.max(axis=axis), axis=axis)
    x_exp = np.exp(x)
    x_exp_row_sum = np.expand_dims(x_exp.sum(axis=axis), axis=axis)
    soft_max = x_exp / x_exp_row_sum

    return soft_max

