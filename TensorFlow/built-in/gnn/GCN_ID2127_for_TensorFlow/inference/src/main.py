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
# Author: Salli Moustafa (salli.moustafa@huawei.com)
"""
GCN inference launching
"""

import argparse
import datetime

import acl

from ats.engine import Engine
from ats.dataset import DataSet
from ats.utils import check_status

from gcn_task import GCNTask


def gcn_inference_execute(model_path,
                          input_path,
                          output_path,
                          device_id,
                          device_count,
                          stream_count,
                          profiling_mode,
                          sparse):
    """
    Prepare and execute GCN inference:
        - start Engine
        - create DataSet
        - create and launch GCN task
    """

    engine = Engine(device_count, stream_count, profiling_mode)

    dataset = DataSet(model_path)

    gcn_task = GCNTask(dataset, input_path, output_path, sparse)

    engine.init(output_path)

    start = datetime.datetime.now()

    engine.launch(gcn_task, device_id)
    engine.wait()

    end = datetime.datetime.now()
    print('GCN.all | {} | {}'.format(device_count, (end - start).total_seconds()))

    engine.finalize()


if __name__ == '__main__':
    status = acl.init("inference/acl.json")
    check_status("acl.rt.set_device", status)

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--sparse', '-sp', default=False, action='store_true',
                        help='Enable sparse API')
    args = parser.parse_args()

    gcn_inference_execute(args.model_path,
                          args.input_path, args.output_path,
                          args.device_id,
                          args.device_count,
                          args.stream_count,
                          args.profiling_mode,
                          args.sparse)

    acl.finalize()
