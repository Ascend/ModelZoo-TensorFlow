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
# ==============================================================================

from npu_bridge.npu_init import *

from config import Config
from train import Train
import os
import shutil
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--output_path", type=str, default="./classifier_result")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_epochs", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--token_len", type=int, default=1000)
parser.add_argument("--config_file", type=str, default="./conf/textcnn_char.config")
args = parser.parse_args()

config = Config(config_file=args.config_file)

config.train.batch_size = args.batch_size
config.train.num_epochs = args.train_epochs
config.train.learning_rate = args.learning_rate
config.var_len_feature.max_var_token_length = args.token_len
config.var_len_feature.max_var_char_length = args.token_len * 2

model_path = "batch_size_{}lr_{}drop_{}_fasttext_v1".format(config.train.batch_size,
                                                            config.train.learning_rate, config.train.hidden_layer_dropout_keep_prob)

Train(config)

f = open('Best.txt')
lines = f.readlines()
f.close()
idx_model = int(lines[-1].split('\t')[1])

files = [str(file_name) for file_name in sorted([int(file) for file in os.listdir('export_model/')])]

path = files[idx_model - 1]

shutil.copytree("export_model/", "{}/model_all/{}".format(args.output_path, model_path))
shutil.copytree("export_model/{}".format(path), "{}/model_save_best/{}".format(args.output_path, model_path))
shutil.copytree("eval_dir/", "{}/result_save/{}".format(args.output_path, model_path))
os.remove('Best.txt')
# clean data
# shutil.rmtree('export_model', ignore_errors=True)
# shutil.rmtree('tfrecord', ignore_errors=True)
# shutil.rmtree('eval_dir', ignore_errors=True)
# shutil.rmtree('checkpoint', ignore_errors=True)
shutil.rmtree('__pycache__', ignore_errors=True)