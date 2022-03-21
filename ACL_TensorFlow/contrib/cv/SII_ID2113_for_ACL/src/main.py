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
# limitations under the License.import os
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
import tensorflow as tf

from solver import Solver
import argparse
import moxing as mox
from datetime import datetime
# # 解析输入参数train_url
# parser = argparse.ArgumentParser()
# parser.add_argument("--train_url", type=str, default="./output")
# config = parser.parse_args()
# 在ModelArts容器创建训练输出目录
import os
current_path = os.path.dirname(os.path.realpath(__file__)) 
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one feed forwrad, default: 256')
tf.flags.DEFINE_string('dataset', 'svhn', 'dataset name for choice [celebA|svhn], default: svhn')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: False')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 64, 'number of sampling images for check generator quality, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    print(os.getcwd())
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
    model_dir = "/svhn/"
    src=current_path+model_dir
    print(src)
    print(os.path.isdir(src))
    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
        model_dir = "/cache/svhn/"
        # 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        mox.file.copy_parallel(model_dir, "obs://liu-ji-hong-teacher/"+cur_time+"/svhn")
#         mox.file.copy_parallel(current_path+"/kernel_meta/", "obs://liu-ji-hong-teacher/semantic-finish/kernel_meta")
    else:
        solver.test()
    

if __name__ == '__main__':
    tf.app.run()
