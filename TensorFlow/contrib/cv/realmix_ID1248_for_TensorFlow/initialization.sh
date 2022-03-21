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
#!/bin/bash
export ML_DATA='/cache/data'
export PYTHONPATH=$PYTHONPATH:/home/ma-user/modelarts/user-job-dir/code
export ASCEND_GLOBAL_LOG_LEVEL=4
export ASCEND_GLOBAL_EVENT_ENABLE=0
python3.7 -m pip install easydict
python3.7 /home/ma-user/modelarts/user-job-dir/code/read_fromobs.py
for seed in 1; do
    for size in 250; do
        python3.7 /home/ma-user/modelarts/user-job-dir/code/scripts/create_split.py --seed=$seed --label_split_size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    wait
done
python3.7 /home/ma-user/modelarts/user-job-dir/code/realmix.py --filters=32 --dataset=cifar10_aug50.1@250-500 --w_match=75 --beta=0.75 --custom_dataset=True --augment=cifar10 --tsa=linear_schedule
python3.7 /home/ma-user/modelarts/user-job-dir/code/realmix.py --filters=32 --dataset=cifar10_aug50.1@250-500 --w_match=75 --beta=0.75 --custom_dataset=True --augment=cifar10 --tsa=linear_schedule --eval_ckpt=True
