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


import json
import os

def read_json(model, filename):
    '''
    Read the json file containing the specified arguments to be loaded
    Args:
        :model:     The model that is being loaded
        :filename:  The filename of the json file to be loaded
    Returns:
        :params:    Json dict of arguments to be loaded
    '''
    if filename != 'none':
        if '.json' not in filename:
            filename = filename+'.json'
        params_file = open(os.path.join('models', model, filename))
        params = json.load(params_file)
        params_file.close()
        return params

    else:
        return {}


def assign_args(args, params, argv):
    '''
    Update args so that each argument specified in the json file is assigned
    Args:
        :args:   The args returned from argparser
        :params: Json dict containing the arguments to assign to args
        :argv:   arguments specified directly by the user
    Returns:
        :args:   args updated with the params from the json dict
    '''
    user_args = [arg[2:] for arg in argv if '--' in arg]
    for argname in params.keys():
        if argname not in user_args:
            (args.__setattr__(argname, params[argname]))

    return args


