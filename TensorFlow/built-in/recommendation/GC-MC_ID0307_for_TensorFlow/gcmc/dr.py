#
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
#
import subprocess
from sklearn.model_selection import ParameterGrid
import random

random.seed(50)

param_grid = {'dr':[0.03,0.4,0.5,0.6,0.7,0.8]}
grid = list(ParameterGrid(param_grid))

for params in grid:
  dr = str(params['dr'])


  #subprocess.check_call(["./train.py", "-d", "ml_100k","--accum",  str(params['ac']),"-do",str(params['dr']),"-nleft", "-nb" , "2", "-e",str(params['epoch']),"--features", "--feat_hidden", str(params['hf']),"--testing" ])
  #subprocess.call("python train.py -d ml_100k --accum " + ac +" -do " + dr + " -nleft -nb 2 -e " + epoch + " --features --feat_hidden "+ hf +" --testing --learning_rate " + lr + " --hidden " + first +" "+ second ,shell=True)
  #subprocess.call("python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden "+ hf +" --testing",shell=True)
  subprocess.call("python train.py -d ml_100k --accum stack -do "+ dr +" -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing", shell=True)
