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
import tensorflow as tf
import os, argparse
import moxing as mox

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/dataset-jcr/电子器件-分类/node/") #obs该目录下放ckpt4个模型文件
parser.add_argument("--train_url", type=str, default="/dataset-jcr/")
args = parser.parse_args()
data_dir = "/cache/dataset"
os.makedirs(data_dir)
mox.file.copy_parallel(args.data_url, data_dir)

outfile = "./checkpoint/node.txt"#需调整
model_path = args.data_url

with tf.Session() as sess:
    tf.train.import_meta_graph(model_path + 'VDSR_adam_epoch_099.ckpt-336400.meta', clear_devices=True)#需调整
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    node_list = [n.name for n in graph_def.node]
    with open(outfile, "w") as f:
        for node in node_list:
            print("node_name", node)
            f.write(node + "\n")

mox.file.copy_parallel("./checkpoint", args.train_url) #obs的output目录得到节点名txt文件