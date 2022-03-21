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

import glob
import importlib


def create_model_object(**kwargs):
    """
    Use model_name to find a matching model class with that name
    All model classes are initialized from the same abstract class so just call that initializer

    Arguments:
    :kwargs: arguments specified in training and testing program

    Returns:
    :model:  model object initialized based off of the given model name
    """

    model_files = glob.glob('models/*/*_model.py')
    all_list    = list()
    model_name  = kwargs['modelName']

    for mf in model_files:
        if 'tsn' not in mf:
            continue
        module_name = mf[:-3]
        module_name = module_name.replace('/','.') # linux
        # module_name = module_name.replace('\\','.') #win

        module = importlib.import_module(module_name)

        # module_lower = map(lambda module_x: module_x.lower(), dir(module))
        module_lower = [i.lower() for i in dir(module)]

        if model_name.lower() in module_lower:
            model_index = module_lower.index(model_name.lower())
            model = getattr(module, dir(module)[model_index])(**kwargs)
            return model

        # END IF

    # END FOR

    print("Model not found, specify model name in lowercase and try again. Ensure model is in a folder within 'models' directory and includes model file 'name_model.py'.")
    exit()
