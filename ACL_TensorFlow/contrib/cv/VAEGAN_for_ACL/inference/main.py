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

from utils import mkdir_p
from vaegan import vaegan
from utils import CelebA

flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 0.0003, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer("repeat", 10000, "the numbers of repeat for your datasets")
flags.DEFINE_string("path", "./Data/img_align_celeba/", "the path of data with attr_txt")
flags.DEFINE_integer("act", "dump", "dump or compare")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    vaegan_checkpoint_dir = "./model/model.ckpt"
    sample_path = "./Data/sample"

    mkdir_p('./model/')
    mkdir_p(sample_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    latent_dim = FLAGS.latent_dim
    data_repeat = FLAGS.repeat

    learn_rate_init = FLAGS.learn_rate_init
    cb_ob = CelebA(FLAGS.path)

    vaeGan = vaegan(batch_size= batch_size, max_iters= max_iters, repeat = data_repeat,
                      model_path= model_path, data_ob= cb_ob, latent_dim= latent_dim,
                      sample_path= sample_path , learnrate_init= learn_rate_init)

    if FLAGS.act == "dump":
        vaeGan.build_model_vaegan()
        vaeGan.dump()

    if FLAGS.act == "compare":
        vaeGan.build_model_vaegan()
        vaeGan.save_acl_image()
        vaeGan.compare_mean()

