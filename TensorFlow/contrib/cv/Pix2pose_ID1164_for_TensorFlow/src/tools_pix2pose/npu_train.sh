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

code_dir=$1
work_dir=$2
dataset_path=$3
output_path=$4

#############训练前输入目录文件确认#########################
echo "[CANN-ZhongZhi] before train - list my run files[/usr/local/Ascend/ascend-toolkit]:"
ls -al /usr/local/Ascend/ascend-toolkit
echo ""

echo "[CANN-ZhongZhi] before train - list my code files[${code_dir}]:"
ls -al ${code_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my dataset files[${dataset_path}]:"
ls -al ${dataset_path}
echo ""

echo "[CANN-ZhongZhi] before train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""

######环境变量修改######
###如果需要修改环境变量的，在此处修改
#设置日志级别为info
#export ASCEND_GLOBAL_LOG_LEVEL=1
#设置日志打屏到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export TF_CPP_MIN_LOG_LEVEL=0
env > ${output_path}/my_env.log
export LD_PRELOAD='/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so'
######数据预处理执行######
#第一步
#执行该命令时，需要将bop_io.py中的 get_dataset函数的两个参数train和eval都置为False
#第一步输入：model_reconst，输出：model_xyz
#第一步预处理需要使用需要使用OpenGL进行喧扰，目前无法在NPU上测试，因此建议在GPU上进行预训练得到结果，再在NPU上进行训练。
#python3.7 ${code_dir}/2_1_ply_file_to_3d_coord_model.py --data_path=${dataset_path} --output_path=${output_path} cfg/cfg_tless_paper.json tless
#第二步
#第二部输入：model_cad/model_eval、输出：train_xyz
#第二步预处理需要使用需要使用OpenGL进行喧扰，目前无法在NPU上测试，因此建议在GPU上进行预训练得到结果，再在NPU上进行训练。
#python3.7 ${code_dir}/2_2_render_pix2pose_training.py --data_path=${dataset_path} --output_path=${output_path} cfg/cfg_tless_paper.json tless
######训练执行######
###训练模型命令，共30个类，--obj_id为类的参数
#训练步骤输入：models_eval、train_xyz、background_dir（train_2017）\train_ primesense 输出：pix2pose_weights
#python3.7 ${code_dir}/3_train_pix2pose.py --data_path=${dataset_path} --output_path=${output_path}  --obj_id='01'
#。。。
#。。。
#python3.7 ${code_dir}/3_train_pix2pose.py --data_path=${dataset_path} --output_path=${output_path}  --obj_id='30'
######测试执行######
#第一步骤
#第一步输入：model_xyz 和pix2pose_weights 和detection_weight 输出：.csv文件
#python3.7 ${code_dir}/5_evaluation_bop_basic.py --data_path=${dataset_path} --output_path=${output_path}
#第二步骤，得出最后结果
#在测试第二步，需要使用OpenGL进行喧扰，目前无法在NPU上测试，可以将测试第一步骤得出的结果在GPU上进行测试
#python3.7  ${code_dir}/scripts/eval_bop19.py --data_path=${dataset_path} --output_path=${output_path} --renderer_type=vispy --result_filenames=pix2pose-iccv19_tless-test-primesense.csv
python3.7  ${code_dir}/scripts/eval_calc_errors.py --n_top=-1 --error_type=vsd --result_filenames=pix2pose-iccv19_tless-test-primesense.csv --renderer_type=vispy --results_path=/home/ma-user/modelarts/inputs/data_url_0/tless/ --eval_path=/home/ma-user/modelarts/outputs/train_url_0/ --targets_filename=test_targets_bop19.json --max_sym_disc_step=0.01 --skip_missing=1 --vsd_deltas=hb:15,icbin:15,icmi:15,itodd:5,lm:15,lmo:15,ruapc:15,tless:15,tudl:15,tyol:15,ycbv:15,hope:15 --vsd_taus=0.05,0.1,0.15000000000000002,0.2,0.25,0.3,0.35000000000000003,0.4,0.45,0.5 --vsd_normalized_by_diameter=True

if [ $? -eq 0 ];
then
    echo "[CANN-ZhongZhi] train return success"
else
    echo "[CANN-ZhongZhi] train return failed"
fi

######训练后把需要备份的内容保存到output_path######
###此处每个网络不同，视情况添加cp
cp -r ${work_dir} ${output_path}

######训练后输出目录文件确认######
echo "[CANN-ZhongZhi] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""
