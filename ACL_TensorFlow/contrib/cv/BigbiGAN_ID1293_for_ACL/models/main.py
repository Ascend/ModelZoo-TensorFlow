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
from absl import app
from absl import flags
import os
from train import set_up_train
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags.DEFINE_string('input_bin_dir', './Bin','Path of the directory for input bin files.')
flags.mark_flag_as_required("input_bin_dir")
flags.DEFINE_string('dataset_path', './tensorflow_datasets','Path for saving dataset.')
flags.mark_flag_as_required("dataset_path")
flags.DEFINE_integer('train_batch_size',256 ,'Batch size for training.')
flags.mark_flag_as_required("train_batch_size")
flags.DEFINE_integer('num_epochs', 50 ,'Number of epochs to train.')    #50
flags.mark_flag_as_required("num_epochs")
flags.DEFINE_string('result_path', './results','Path for saving results.')
flags.mark_flag_as_required("result_path")
flags.DEFINE_string('output_pb_path', './results/pb','Path for saving pb file.')
flags.mark_flag_as_required("output_pb_path")
flags.DEFINE_string('output_gpu_generated_img', './results/generated_images','Path for saving generated image generated by gpu.')
flags.mark_flag_as_required("output_gpu_generated_img")

flags.DEFINE_string("model", 'BigBiGAN', 'Model to use (BigBiGAN|')
flags.DEFINE_string("dataset",'mnist','Dataset (mnist|fashion_mnist|cifar10)')
flags.DEFINE_integer('num_classes', 10, 'Numbers of classes in the dataset')
flags.DEFINE_integer('logging_step',100,'Step number for logging')
flags.DEFINE_integer('data_buffer_size',1000,'Buffersize input pipeline.')
flags.DEFINE_bool('cache_dataset', False, 'cache dataset (True|False).')
flags.DEFINE_string('device', 'GPU', 'Device using now(CPU|GPU)')
flags.DEFINE_integer('gen_disc_ch',64,'Number of channels in the first layer of generator and discriminator_f.')
flags.DEFINE_integer('en_ch',32,'Number of channels in the first layer of encoder.')
flags.DEFINE_float('lr_gen_en',2e-4,'Learning rate generator.')
flags.DEFINE_float('beta_1_gen_en',0.5,'Beta_1 of Generator optimizer.')
flags.DEFINE_float('beta_2_gen_en',0.999,'Beta_2 of generator optimizer.')
flags.DEFINE_float('lr_disc',2e-4,'Learning rate discriminator.')
flags.DEFINE_float('beta_1_disc',0.5,'Beta_1 of Discriminator optimizer.')
flags.DEFINE_float('beta_2_disc',0.999,'Beta_2 of discriminator optimizer.')
flags.DEFINE_integer('D_G_ratio', 2, 'Ratio of upgrading weights, discriminator VS generator & encoder')
flags.DEFINE_integer('num_cont_noise', 100, 'Dimension of continous noise vector.')
flags.DEFINE_bool('conditional', True, 'Conditional or unconditional GAN')
flags.DEFINE_integer('num_emb', 32, 'Dimension of embedded label output. Only applicable when conditional')


def main(argv):
    del argv  # Unused.
    set_up_train(FLAGS)


if __name__ == '__main__':
    app.run(main)
