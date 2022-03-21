# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

import datetime
import json
import os
import subprocess

from absl import flags

from libs.convert_om import convert_om

FLAGS = flags.FLAGS


def npu_predict(inputs):
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)

    if os.path.exists(output_dir):
        cmd = "rm -f %s/*.bin" % output_dir
        subprocess.call(cmd, shell=True, stderr=subprocess.PIPE)
    else:
        os.makedirs(output_dir)

    model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
    output_pre = os.path.join(output_dir, model_name)
    device_id = os.getenv("DEVICE_ID")

    if device_id is None:
        device_id = "0"

    dump_param = ""

    if FLAGS.merge_input:
        cmd = "./xacl_fmk -m %s -i %s -o %s -b %s -g 1 -n %s %s" % \
              (FLAGS.om_model_file, inputs, output_pre, FLAGS.predict_batch_size, device_id, dump_param)
    else:
        cmd = "./xacl_fmk -m %s -i %s -o %s -b %s -n %s %s" % \
              (FLAGS.om_model_file, inputs, output_pre, FLAGS.predict_batch_size, device_id, dump_param)

    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                    "I", "ACL cmd: %s" % cmd))
    subprocess.call(cmd, shell=True, stderr=subprocess.PIPE)
    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                    "I", "ACL cmd finished"))
