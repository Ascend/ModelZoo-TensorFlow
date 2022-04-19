# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from itertools import chain
import numpy as np
from yacs.config import CfgNode, _VALID_TYPES, _assert_with_logging, _valid_type


def to_pair(x, num_reps):
    """Make paired of the int.

    Args:
        x: int
        num_reps: int, number of replicate of the input x.
    
    Return:
        list[int], where each value is a copy of the input x.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        assert len(x) == num_reps
    elif isinstance(x, int):
        x = [x] * num_reps
    else:
        raise ValueError
    return x


def convert_to_dict(cfg_node, key_list):
    """Convert yacs node to dict.

    Usage:
        # a is a yacs node
        a_as_dict = convert_to_dict(a, [])

    Args:
        cfg_node: a yacs node.
        key_list: list[str], the key name in the dict.

    Return:
        dict, a dict version of the config node.
    """
    if not isinstance(cfg_node, CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def convert_dict_to_list(cfg_dict, prefix_key=None):
    """Convert a dict to list
    """
    if prefix_key and prefix_key == '':
        prefix_key = None

    cfg_list = []
    
    for k, v in cfg_dict.items():
        cur_key = f'{prefix_key}.{k}' if prefix_key is not None else k
        if isinstance(v, dict):
            cfg_list_sub = convert_dict_to_list(v, prefix_key=cur_key)
            cfg_list.extend(cfg_list_sub)
            # new_keys = [f'{k}.{sub_k}' for sub_k in cfg_list_sub[::2]]
            # values = cfg_list_sub[1::2]
            # cfg_list.extend(list(chain(*zip(new_keys, values))))
        elif isinstance(v, (list, tuple)):
            cfg_list.extend([cur_key, f'{v}'])
        else:
            cfg_list.extend([cur_key, v])
            # cfg_list.extend([cur_key, f'{v}'])
    return cfg_list
