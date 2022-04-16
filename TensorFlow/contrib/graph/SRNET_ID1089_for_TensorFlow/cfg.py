"""
SRNet - Editing Text in the Wild
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

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

import os
import argparse
import moxing as mox

# pretrained vgg
vgg19_weights = r'/home/ma-user/modelarts/user-job-dir/code/model_logs/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.pb'

# model parameters
lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-4 # default 1e-3
decay_rate = 0.9
decay_steps = 10000
staircase = False
beta1 = 0.9 # default 0.9
beta2 = 0.999 # default 0.999
max_iter = 500000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 10000
gen_example_interval = 1000
checkpoint_savedir = '/cache/out/model_logs/checkpoints'
tensorboard_dir = '/cache/out/model_logs/train_logs'
pretrained_ckpt_path = None
train_name = None # used for name examples and tensorboard logdirs, set None to use time

# data
batch_size = 8
data_shape = [64, None]
data_dir = '/cache/data'
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
example_data_dir = r'/home/ma-user/modelarts/user-job-dir/code/examples/labels'
example_result_dir = '/cache/out/genLogs'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = 'examples/result'

#out_dir = ''

def main():
    code_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str)
    parser.add_argument("--data_url", type=str)
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/data")
    parser.add_argument("--modelarts_output_dir", type=str, default="/cache/out")
    paraConfig = parser.parse_args()

    #out_dir = paraConfig.modelarts_output_dir
    os.makedirs(paraConfig.modelarts_data_dir)
    os.makedirs(paraConfig.modelarts_output_dir)
    os.makedirs(r'/cache/out/genLogs')
    mox.file.copy_parallel(src_url=paraConfig.data_url, dst_url=paraConfig.modelarts_data_dir)
    os.system("python3" + ' ' + code_dir + '/train.py')

if __name__ == "__main__":
    main()
