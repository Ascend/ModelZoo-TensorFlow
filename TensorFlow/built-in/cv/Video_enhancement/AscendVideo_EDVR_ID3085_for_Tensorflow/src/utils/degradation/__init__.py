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
from .noise import *
from .scaling import *
from .blur import *

from src.utils.klass import get_subclass_given_name


class Degradation(object):
    """Composes several degradations together.

    Args:
        transforms: list[Transform], list of joint transforms to compose.
    """
    def __init__(self, degradations=None):
        self.degradations = degradations

    @classmethod
    def from_cfgs(cls, options, **kwargs):
        """Construct augmentation pipeline from cfg dict.

        Args:
            options: dict, pairs of {Transform_class_type: kwargs}.
            kwargs: dict, additional kwargs.

        Returns:
            A composed transform instance.
        """
        
        t = []
        for k, v in options.items():
            if k == 'RandomCrop':
                # crop_size and scales are required terms
                v['crop_size'] = kwargs['crop_size']
                v['scales'] = kwargs['scales']
            elif k == 'Scaling':
                v['scales'] = kwargs['scales']
            _filter = get_subclass_given_name(Base, k)
            t.append(_filter(**v))
        return cls(t)

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
