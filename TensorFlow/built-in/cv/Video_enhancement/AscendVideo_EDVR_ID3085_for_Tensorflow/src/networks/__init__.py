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
import inspect
import importlib

from .base_model import Base
from .register import registry

model_dir = os.path.dirname(os.path.realpath(__file__))
model_file = os.listdir(model_dir)

# Automatically import all the defined class in the files under src.networks
__all__ = ['registry']
for model in model_file:
    module_name = model.split('.')[0]
    if module_name in ['register', 'base_model', '__init__', 'VSR']:
        continue
    mod = importlib.import_module(f'.{module_name}', 'src.networks')
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Base) and module_name not in __all__ and obj.__module__ == module_name:
            __all__.append(name)


def build_network(cfg):
    network = registry[cfg.model.name](cfg=cfg)
    return network


__all__.append('build_network')
