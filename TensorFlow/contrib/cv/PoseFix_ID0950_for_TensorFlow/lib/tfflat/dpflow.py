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


from npu_bridge.npu_init import *
import zmq
import multiprocessing as mp
from .serialize import loads, dumps

def data_sender(id, name, func_iter, *args):
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect('ipc://@{}'.format(name))

    print('start data provider {}-{}'.format(name, id))
    while True:
        data_iter = func_iter(id, *args)
        for msg in data_iter:
            # print(id)
            sender.send( dumps([id, msg]) )

def provider(nr_proc, name, func_iter, *args):
    proc_ids = [i for i in range(nr_proc)]

    procs = []
    for i in range(nr_proc):
        w = mp.Process(target=data_sender, args=(proc_ids[i], name, func_iter, *args))
        w.deamon = True
        procs.append( w )

    for p in procs:
        p.start()

def receiver(name):
    context = zmq.Context()

    receiver = context.socket(zmq.PULL)
    receiver.bind('ipc://@{}'.format(name))

    while True:
        id, msg = loads( receiver.recv() )
        # print(id, end='')
        yield msg

