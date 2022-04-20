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


def get_subclass_given_name(proto_type, sub_type_name):
    """Get the subclass type given the name

    Look through all subclasses and select by type name.
    
    Args:
        proto_type: base class.
        sub_type_name: str, derived class name.

    Returns:
        derived class type.
    """
    subtype = [
        stp for stp in proto_type.__subclasses__()
        if stp.__name__ == sub_type_name
    ]
    return subtype[0]


class Singleton(type):
    """Singleton class type.

    A singleton class will only have one instance.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
