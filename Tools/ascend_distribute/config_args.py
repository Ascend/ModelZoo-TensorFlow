#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
import os
import yaml

from enum import Enum
from dataclasses import dataclass


cache_home = os.path.expanduser("~/ascend_tools")
cache_dir = os.path.join(cache_home, "ascend_distribute/")
library_path = os.path.join(cache_dir, "custom_tune_bank")
if not os.path.exists(library_path):
    os.makedirs(library_path)
listnames = os.listdir(library_path)
bank_path=None
if listnames:
    for names in listnames:
        if "gradient_fusion" in names:
            bank_path = os.path.join(library_path, names)
default_config_file = os.path.join(cache_dir, "ascend_distribute_config.yaml")
default_rank_table_file = os.path.join(cache_dir, "rank_table.json")


def load_config_from_file(config_file):
    config_file_exists = config_file is not None and os.path.isfile(config_file)
    config_file = config_file if config_file_exists else default_config_file

    with open(config_file, "r", encoding="utf-8") as f:
        config_class = MultiWorkerConfig
        return config_class.from_yaml_file(yaml_file=config_file)


@dataclass
class BaseConfig:
    device_id: int

    def to_dict(self):
        result = self.__dict__

        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        return result

    def to_yaml_file(self, yaml_file):
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)

    @classmethod
    def from_yaml_file(cls, yaml_file=None):
        yaml_file = default_config_file if yaml_file is None else yaml_file
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        print(config_dict)
        if "start_rank_id" in config_dict:
            config_dict['multi_worker'] = True
        return cls(**config_dict)

    def __post_init__(self):
        pass


@dataclass
class MultiDeviceConfig(BaseConfig):
    rank_nums: int
    device_list: list
    rank_size: int = 8

    def __post_init__(self):
        return super().__post_init__()


@dataclass
class MultiWorkerConfig(BaseConfig):
    start_rank_id: int = 0
    multi_worker: bool = False
    worker_num: int = 1

    def __post_init__(self):
        return super().__post_init__()