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

from core.leras import nn

nn.initialize_main_env()
import os
import sys
import time
import argparse

from core import pathex
from core import osex
from pathlib import Path
from core.interact import interact as io
# from npu_bridge.npu_init import *


p = argparse.ArgumentParser()

p.add_argument('--training_data_src_dir', default='',
               help="Dir of extracted SRC faceset.")
p.add_argument('--training_data_dst_dir', default='',
               help="Dir of extracted DST faceset.")

p.add_argument('--input_dir', dest="input_dir", default="",
               help="Input directory. A directory containing the files you wish to process.")
p.add_argument('--output_dir', dest="output_dir", default="",
               help="Output directory. This is where the merged files will be stored.")
p.add_argument('--output_mask_dir', dest="output_mask_dir", default="",
               help="Output mask directory. This is where the mask files will be stored.")

p.add_argument('--model_dir', default='', help="Saved models dir.")
p.add_argument('--model_class_name', default='Quick96', help="Model class name.")


p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
p.add_argument('--no_preview', action="store_false", default=True, help="Disable preview window.")
p.add_argument('--force_model_name', dest="force_model_name", default='Trained',
               help="Forcing to choose model name from model/ folder.")
p.add_argument('--cpu_only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
p.add_argument('--force_gpu_idxs', dest="force_gpu_idxs", default='0',
               help="Force to choose GPU indexes separated by comma.")
p.add_argument('--silent_start', action="store_true", dest="silent_start", default=False,
               help="Silent start. Automatically chooses Best GPU and last used model.")
p.add_argument('--execute_program', dest="execute_program", default=[], action='append', nargs='+')
p.add_argument('--target_iters', type=int, default=20000)
p.add_argument('--eval_iters', type=int, default=1000)
p.add_argument('--stop_SSIM', type=float, default=0.66)
arguments = p.parse_args()


def process_train():
    kwargs = {'training_data_src_path': Path(arguments.training_data_src_dir),
              'training_data_dst_path': Path(arguments.training_data_dst_dir),
              'saved_models_path': Path(arguments.model_dir),
              'model_class_name': arguments.model_class_name,

              'input_path': Path(arguments.input_dir),
              'output_path': Path(arguments.output_dir),
              'output_mask_path': Path(arguments.output_mask_dir),

              'no_preview': arguments.no_preview,
              'force_model_name': arguments.force_model_name,
              'force_gpu_idxs': [int(x) for x in arguments.force_gpu_idxs.split(
                  ',')] if arguments.force_gpu_idxs is not None else None,
              'cpu_only': arguments.cpu_only,
              'silent_start': arguments.silent_start,
              'execute_programs': [[int(x[0]), x[1]] for x in arguments.execute_program],
              'debug': arguments.debug,
              'target_iters': arguments.target_iters,
              'eval_iters': arguments.eval_iters,
              'stop_SSIM': arguments.stop_SSIM,
              }
    from mainscripts import Trainer
    Trainer.main(**kwargs)


if __name__ == "__main__":
    exit_code = 0
    print("===========model train start============")
    process_train()
    print("===========model train end============")

    if exit_code == 0:
        print("Done.")

    exit(exit_code)