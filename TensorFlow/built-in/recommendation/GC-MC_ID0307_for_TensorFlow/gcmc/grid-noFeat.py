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

param_grid = {'epoch' :[1000, 100, 50,500], 'lr':[0.0005, 0.001,0.00146,0.01,0.015,0.0175,0.02], 'dr':[0.03,0.4,0.5,0.6,0.7,0.8],'hf':[5,10,15], 'ac':['stack','sum'], 'hn':[(50,8),(10,7),(50,3),(100,25),(40,15),(500,75),(45,15)]}
#param_grid = {'hf':[5,10,15]}
grid = list(ParameterGrid(param_grid))

e_val = random.sample(grid, 100)
#print(e_val)

for params in e_val:
  epoch = str(params['epoch'])
  #e_val = random.sample(params['epoch'],1)[0]
  ac = str(params['ac'])
  dr = str(params['dr'])
  lr = str(params['lr'])
  hf = str(params['hf'])
  x = params['hn']
  first = str(x[0])
  second = str(x[1])

  #subprocess.check_call(["./train.py", "-d", "ml_100k","--accum",  str(params['ac']),"-do",str(params['dr']),"-nleft", "-nb" , "2", "-e",str(params['epoch']),"--features", "--feat_hidden", str(params['hf']),"--testing" ])
  subprocess.call("python train.py -d ml_100k --accum " + ac +" -do " + dr + " -nleft -nb 2 -e " + epoch + " --testing --learning_rate " + lr + " --hidden " + first +" "+ second ,shell=True)
  #subprocess.call("python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden "+ hf +" --testing",shell=True)
