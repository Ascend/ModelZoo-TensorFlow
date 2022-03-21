#!/usr/bin/env python3
#encoding: UTF-8
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
#File: rrc_evaluation_funcs_1_1.py
#Version: 1.1
#Version info: changes for Python 3
#Date: 2019-12-29
#Description: File with useful functions to use by the evaluation scripts in the RRC website.

import tensorflow as tf
tf.app.flags.DEFINE_string('conf_path',"", '')
tf.app.flags.DEFINE_integer('npu_nums', 0, '')
FLAGS = tf.app.flags.FLAGS


import json
import os 
server=[]
server_conf=[]
server_list=["0","1","2","3","4","5","6","7"]
if os.path.isdir(FLAGS.conf_path):
    for f in os.listdir(FLAGS.conf_path):
        if (f.split("_")[-1]).split(".")[0] in server_list and (f.split("_")[-1]).split(".")[1] == 'info' and f.split("_")[0]=='server':
            server_conf.append(f)                 #将寻找到的文件添加到server_conf列表中


rank_address=[]
for i in range(FLAGS.npu_nums):
    for x in server_conf:
        if (x.split("_")[-1]).split(".")[0] == str(i):
            server.append(x.split("_")[1])
            l = FLAGS.conf_path + "/" + x
            with open(l, "r") as a:
                s = a.readlines()
                for s_ in s:
                    if 'address' in s_:
                        rank_address.append(s_.split("=")[-1][:-1])

rank_16p = {
  "server_count":"2",
  "server_list":[
    {
        "server_id":server[0],
      "device":[
        {
          "device_id":"0",
          "device_ip":rank_address[0],
          "rank_id":"0"
        },
        {
          "device_id":"1",
          "device_ip":rank_address[1],
          "rank_id":"1"
        },
        {
          "device_id":"2",
          "device_ip":rank_address[2],
          "rank_id":"2"
        },
        {
          "device_id":"3",
          "device_ip":rank_address[3],
          "rank_id":"3"
        },
        {
          "device_id":"4",
          "device_ip":rank_address[4],
          "rank_id":"4"
        },
        {
          "device_id":"5",
          "device_ip":rank_address[5],
          "rank_id":"5"
        },
        {
          "device_id":"6",
          "device_ip":rank_address[6],
          "rank_id":"6"
        },
        {
          "device_id":"7",
          "device_ip":rank_address[7],
          "rank_id":"7"
        }
      ]},


      {
      "server_id":server[1],
      "device":[
        {
          "device_id":"0",
          "device_ip":rank_address[8],
          "rank_id":"8"
        },
        {
          "device_id":"1",
          "device_ip":rank_address[9],
          "rank_id":"9"
        },
        {
          "device_id":"2",
          "device_ip":rank_address[10],
          "rank_id":"10"
        },
        {
          "device_id":"3",
          "device_ip":rank_address[11],
          "rank_id":"11"
        },
        {
          "device_id":"4",
          "device_ip":rank_address[12],
          "rank_id":"12"
        },
        {
          "device_id":"5",
          "device_ip":rank_address[13],
          "rank_id":"13"
        },
        {
          "device_id":"6",
          "device_ip":rank_address[14],
          "rank_id":"14"
        },
        {
          "device_id":"7",
          "device_ip":rank_address[15],
          "rank_id":"15"
        }
      ]
    }
  ],
  "status":"completed",
  "version":"1.0"
}


with open ("rank_table.json","w") as f:
    json.dump(rank_16p, f)






















print(server)
print(rank_address)

