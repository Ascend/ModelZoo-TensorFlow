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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ThinPlateSplineB import ThinPlateSpline3 as TPS
import tensorflow as tf
__all__ = [tf]
import argparse
import numpy as np
# ***************************************Train mode*********************************************************

# Parameter setting ****************************************************************************************************
MODE = "train"
LANDMARK_N = 8
CHECKPOINT = None

SAVE_FREQ = 500
SUMMARY_FREQ = 20
BATCH_SIZE = 1
DOWNSAMPLE_M = 4
DIVERSITY = 500.
ALIGN = 1.
LEARNING_RATE = 1.e-4
MOMENTUM = 0.5
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0005
SCALE_SIZE = 146
CROP_SIZE = 146
MAX_EPOCH = 200

def feature_warping2(feature, deformation, padding=0):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
    ])
    feature = tf.pad(feature, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    CROP_SIZE = feature.get_shape()[1]
    Batch_SIZE = feature.get_shape()[0]
    DEPTH = feature.get_shape()[3]

    grid = tf.constant(t_.reshape([1, 4, 2]), dtype=tf.float32)
    grid = tf.tile(grid, [Batch_SIZE, 1, 1])

    input_images_expanded = tf.reshape(feature, [Batch_SIZE, CROP_SIZE, CROP_SIZE, DEPTH, 1])
    t_img = TPS(input_images_expanded, grid, deformation, [CROP_SIZE, CROP_SIZE, DEPTH])
    t_img = tf.reshape(t_img, tf.shape(feature))
    t_img = tf.image.crop_to_bounding_box(t_img, padding, padding, CROP_SIZE-2*padding, CROP_SIZE-2*padding)
    return t_img

def align_loss2(predA, predB, deformation, n_landmarks):


    # compute the mean of landmark locations

    batch_size = predA.get_shape()[0]
    pred_size = predA.get_shape()[1]
    index = tf.range(0, tf.cast(pred_size, tf.float32), delta=1, dtype=tf.float32)
    index = tf.reshape(index, [pred_size, 1])

    x_index = tf.tile(index, [1, pred_size])

    index = tf.transpose(index)

    y_index = tf.tile(index, [pred_size, 1])

    x_index = tf.expand_dims(x_index, 2)
    x_index = tf.expand_dims(x_index, 0)

    y_index = tf.expand_dims(y_index, 2)
    y_index = tf.expand_dims(y_index, 0)

    x_index = tf.tile(x_index, [batch_size, 1, 1, n_landmarks])
    y_index = tf.tile(y_index, [batch_size, 1, 1, n_landmarks])


    u_norm2 = tf.pow(x_index, 2.) + tf.pow(y_index, 2.)
    u_norm2 = u_norm2 * predA
    loss_part1 = tf.reduce_sum(u_norm2, axis=[1, 2])

    x_index_deformed = feature_warping2(x_index, deformation, padding=3)
    y_index_defomred = feature_warping2(y_index, deformation, padding=3)
    v_norm2 = tf.pow(x_index_deformed, 2.) + tf.pow(y_index_defomred, 2.)
    v_norm2 = v_norm2 * predB
    loss_part2 = tf.reduce_sum(v_norm2, axis=[1, 2])


    loss_part3x = tf.reduce_sum(x_index * predA, axis=[1, 2])
    loss_part3y = tf.reduce_sum(y_index * predA, axis=[1, 2])
    loss_part4x = tf.reduce_sum(x_index_deformed * predB, axis=[1, 2])
    loss_part4y = tf.reduce_sum(y_index_defomred * predB, axis=[1, 2])

    loss_part3 = loss_part3x * loss_part4x + loss_part3y * loss_part4y
    loss = loss_part1 + loss_part2 - 2. * loss_part3
    loss = tf.reduce_mean(loss)

    return loss




def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Factorized Spatial Embeddings")
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--K", type=int, default=LANDMARK_N,
                        help="Number of landmarks.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for adam.")
    parser.add_argument("--beta1", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--M", type=int, default=DOWNSAMPLE_M,
                        help="Downsampling value of the diversity loss.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--diversity_weight", type=float, default=DIVERSITY,
                        help="Weight on diversity loss.")
    parser.add_argument("--align_weight", type=float, default=ALIGN,
                        help="Weight on align loss.")
    parser.add_argument("--scale_size", type=int, default=SCALE_SIZE,
                        help="Scale images to this size before cropping to CROP_SIZE")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE,
                        help="CROP images to this size")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCH,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--summary_freq", type=int, default=SUMMARY_FREQ,
                        help="Update summaries every summary_freq steps")
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ, help="Save model every save_freq steps")
    parser.add_argument("--num_gpus", default=1)
    return parser.parse_args()
def main():
    """Create the model and start the training."""
    deformationlog = np.load("deformationlog.npy")
    deformation= tf.convert_to_tensor(deformationlog)


    args = get_arguments()
    tf.set_random_seed(args.random_seed)
    predA = np.loadtxt("new0_output_0.txt")
    predB = np.loadtxt("new0_output_1.txt")

    predA = tf.convert_to_tensor(predA, dtype = tf.float32)
    predB = tf.convert_to_tensor(predB, dtype = tf.float32)
    # apply a spatial softmax to obtain K probability maps

    pred_size = 42
    predA = tf.reshape(predA, [-1, pred_size * pred_size, args.K])
    predB = tf.reshape(predB, [-1, pred_size * pred_size, args.K])

    predA = tf.nn.softmax(predA, dim=1)  
    predB = tf.nn.softmax(predB, dim=1)  

    predA = tf.reshape(predA, [-1, pred_size, pred_size, args.K])
    predB = tf.reshape(predB, [-1, pred_size, pred_size, args.K])

    algn_loss = align_loss2(predA, predB, deformation, n_landmarks= args.K)
    with tf.Session() as sess:
            print('algn_loss',sess.run(algn_loss))


if __name__ == '__main__':
    main()
