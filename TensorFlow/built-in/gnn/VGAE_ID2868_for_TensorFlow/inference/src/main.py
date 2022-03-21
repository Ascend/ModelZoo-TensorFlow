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
# Author: Salli Moustafa 
"""
VGAE inference launching
"""

import argparse
import datetime

import acl

from ats.engine import Engine
from ats.dataset import DataSet
from ats.utils import check_status

from vgae_task import VGAETask


def vgae_inference_execute(dataset_name,
                           model_path,
                           input_path,
                           output_path,
                           device_id,
                           device_count,
                           stream_count,
                           profiling_mode):
    """
    Prepare and execute VGAE inference:
        - start Engine
        - create DataSet
        - create and launch VGAE task
    """

    engine = Engine(device_count, stream_count, profiling_mode)

    dataset = DataSet(model_path, dataset_name)

    vgae_task = VGAETask(dataset, input_path, output_path)

    engine.init(output_path)

    start = datetime.datetime.now()

    engine.launch(vgae_task, device_id)
    engine.wait()

    end = datetime.datetime.now()
    print('VGAE.all | {} | {}'.format(device_count, (end - start).total_seconds()))

    engine.finalize()


if __name__ == '__main__':
    status = acl.init("inference/acl.json")
    check_status("acl.rt.set_device", status)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-ds',
                        type=str, help='Dataset name')
    parser.add_argument('--model_path', '-mp',
                        type=str, help='Full path to the offline model')
    parser.add_argument('--input_path', '-input',
                        type=str, help='Full path to the input data')
    parser.add_argument('--output_path', '-output',
                        type=str, help='Full path to the output data')
    parser.add_argument('--device_id', '-d', default=0,
                        type=int, help='target device id')
    parser.add_argument('--device_count', '-dc', default='1',
                        type=int, help='device count')
    parser.add_argument('--stream_count', '-sc', default='1',
                        type=int, help='stream count per device')
    parser.add_argument('--profiling_mode', '-pf', default=False, action='store_true',
                        help='Enable profiling')
    args = parser.parse_args()

    vgae_inference_execute(args.dataset_name,
                           args.model_path,
                           args.input_path, args.output_path,
                           args.device_id,
                           args.device_count,
                           args.stream_count,
                           args.profiling_mode)

    acl.finalize()
