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

code_dir=$1
work_dir=$2
dataset_path=$3
output_path=$4

############# Confirm the input directories and files before training #########################
echo "[CANN-Modelzoo] before train - list my run files[/usr/local/Ascend/ascend-toolkit]:"
ls -al /usr/local/Ascend/ascend-toolkit
echo ""

echo "[CANN-Modelzoo] before train - list my code files[${code_dir}]:"
ls -al ${code_dir}
echo ""

echo "[CANN-Modelzoo] before train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-Modelzoo] before train - list my dataset files[${dataset_path}]:"
ls -al ${dataset_path}
echo ""

echo "[CANN-Modelzoo] before train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""

###### Environment variable modification######
### Modify environment variables here if needed
# Set the log level to info
#export ASCEND_GLOBAL_LOG_LEVEL=1
# Set the log print to the screen
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export TF_CPP_MIN_LOG_LEVEL=0

env >${output_path}/my_env.log

###### Training execution ######
### Modify according to the commands to run
python3.7 ${code_dir}/trn_HDSLR_parallel_mri.py --data_path=${dataset_path} --output_path=${output_path} --steps=100
if [ $? -eq 0 ]; then
  echo "[CANN-Modelzoo] train return success"
else
  echo "[CANN-Modelzoo] train return failed"
fi

###### Save the contents to output_path ######
### Add cp if needed
cp -r ${work_dir} ${output_path}

###### Print directory and file confirmation after training ######
echo "[CANN-Modelzoo] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-Modelzoo] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""
