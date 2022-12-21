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
# ==============================================================================

from rediscluster import StrictRedisCluster
import sys
import json
import numpy as np


def redis_conn():
    redis_nodes = [{'host': 'xx', 'port': 5555},
                   {'host': 'xx', 'port': 5555},
                   {'host': 'xx', 'port': 5555},
                   {'host': 'xx', 'port': 5555},
                   {'host': 'xx', 'port': 5555}]
    try:
        redisconn = StrictRedisCluster(startup_nodes=redis_nodes)
    except Exception as e:
        print('Connect Error, %s' % (e,))
        sys.exit()
    return redisconn


def wrap_key(key):
    return 'we:v0:{%s}' % (key,)


def cosine(x1, x2):
    v1, v2 = np.array(x1), np.array(x2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))


if __name__ == '__main__':

    args = sys.argv[1:]
    if not args:
        print('No words.')
        exit()

    _redis = redis_conn()
    if len(args) == 1:
        word = args[0]
        res = _redis.get(wrap_key(word))
        if not res:
            print('can not find word: %s' % (word,))
        print(json.loads(res))
    elif len(args) == 2:
        w1, w2 = args[0], args[1]
        x1, x2 = json.loads(_redis.get(wrap_key(w1))), json.loads(_redis.get(wrap_key(w2)))
        cos_value = cosine(x1, x2)
        print(cos_value)
    else:
        print('More than 2 words.')
