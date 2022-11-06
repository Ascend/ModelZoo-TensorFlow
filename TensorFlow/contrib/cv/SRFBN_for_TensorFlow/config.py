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
import os

class config:
    def __init__(self):
        self.batchsize = 1
        self.Process_num = 3 
        self.maxsize = 200 
        self.ngpu = 1 
        self.imagesize = 64
        self.scale = 3
        self.epoch = 1000
        #create ckpt,log,result dir
        self.checkpoint_dir = "./model"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.log_dir = "./log"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.result = "./result"
        if not os.path.exists(self.result):
            os.mkdir(self.result)



class SRFBN_config(config):
    def __init__(self):
        super(SRFBN_config, self).__init__()
        self.istrain = True#is train or is test
        self.istest = not self.istrain
        self.c_dim = 3 #color channel can train one-channel pic or RGB pic
        self.in_channels = 3
        self.out_channels = 3
        self.num_features = 32#base number of filter
        self.num_steps = 4# timestep
        self.num_groups = 6#the number of projection group of FBB feedbackblock 
        self.BN = True
        if self.BN:
            self.BN_type = "BN" # "BN" # or "IN"
        self.act_type = "prelu" #activation function
        self.loss_type = "L2"
        self.lr_steps = [150, 300, 550, 750]#iteration 
        self.lr_gama = 1
        self.learning_rate = 2e-7#learning rate
        self.load_premodel = True 
        #create dir
        self.srfbn_logdir = "%s/srfbn" % self.log_dir
        if not os.path.exists(self.srfbn_logdir):
            os.mkdir(self.srfbn_logdir)
        self.srfbn_result = "%s/srfbn" % self.result
        if not os.path.exists(self.srfbn_result):
            os.mkdir(self.srfbn_result)

