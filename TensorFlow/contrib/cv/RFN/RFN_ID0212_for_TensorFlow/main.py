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
import sys
import time
import numpy as np
import utils1
import argparse
import tensorflow as tf
from models.RFN import RFN
import logging

parser = argparse.ArgumentParser(description="RFN")
parser.add_argument("--batch_size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--lr_dir", type=str, default=None)
parser.add_argument("--hr_dir", type=str, default=None)
parser.add_argument("--save_checkpoints_steps", type=int, default=10000,
                    help="output patch size")
parser.add_argument("--save_summary_steps", type=int, default=10000,
                    help="output patch size")
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--mini_steps", type=int, default=100000)
parser.add_argument("--stage", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lr_decay", type=float, default=0.7)
parser.add_argument("--sync_replicas", type=int, default=-1)
parser.add_argument("--gpu", type=str, default='0')


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

'''
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
 
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
 
# create file handler which logs even debug messages
fh = logging.FileHandler('../log_out/tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
'''


def create_input_fn(batch_size):
    """The input function of estimator"""

    if(not os.path.exists(args.lr_dir) or
       not os.path.exists(args.hr_dir)):
        raise IOError("Training dataset not found")

    def input_fn():
        """Get input from dataset"""

        def parser(lr_imgs, hr_imgs):
            """Read image from the file system"""

            lr_imgs = tf.image.decode_png(tf.read_file(lr_imgs), channels=3)
            hr_imgs = tf.image.decode_png(tf.read_file(hr_imgs), channels=3)

            lr_imgs = tf.div(tf.to_float(lr_imgs), 255)
            hr_imgs = tf.div(tf.to_float(hr_imgs), 255)

            return lr_imgs, hr_imgs

        def get_patch(lr_imgs, hr_imgs):
            """Get a random patch of image"""

            shape = tf.shape(lr_imgs)
            lh = shape[0]
            lw = shape[1]
            scale = args.scale
            patch_size = args.patch_size // scale

            lx = tf.random_uniform(shape=[1],
                                   minval=0,
                                   maxval=lw - patch_size + 1,
                                   dtype=tf.int32)[0]
            ly = tf.random_uniform(shape=[1],
                                   minval=0,
                                   maxval=lh - patch_size + 1,
                                   dtype=tf.int32)[0]
            hx = lx * scale
            hy = ly * scale

            lr_patch = lr_imgs[ly:ly + patch_size,
                               lx:lx + patch_size]
            hr_patch = hr_imgs[hy:hy + patch_size * scale,
                               hx:hx + args.patch_size]

            return lr_patch, hr_patch

        hr_imgs = sorted(os.listdir(args.hr_dir))
        lr_imgs = sorted(os.listdir(args.lr_dir))
        hr_imgs = [os.path.join(args.hr_dir, img) for img in hr_imgs]
        lr_imgs = [os.path.join(args.lr_dir, img) for img in lr_imgs]
        img_pair = (lr_imgs, hr_imgs)

        dataset = tf.data.Dataset.from_tensor_slices(img_pair)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.map(get_patch, num_parallel_calls=4)
        dataset = dataset.shuffle(32).repeat().batch(
            batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=256)

        return dataset.make_one_shot_iterator().get_next(), None

    return input_fn


def model_fn(features, labels, mode, hparams):
    """The model function of estimator"""
    
    loss_l1 = 0
    loss_l2 = 0

    labels = features[1]
    with tf.variable_scope("RFN"):
        sr_img = RFN(features[0], nf=64, nb=24, out_nc=3)
        loss_l1 += tf.reduce_mean(tf.abs(sr_img - labels))
        loss_l2 += tf.reduce_mean(tf.square(sr_img - labels))

    sr_img = tf.round(tf.clip_by_value(sr_img, 0, 1.0) * 255)
    labels = tf.round(tf.clip_by_value(labels, 0, 1.0) * 255)
    psnr = tf.image.psnr(sr_img, labels, max_val=255)

    loss = loss_l1 / 2 + loss_l2 / 2

    return {
        "loss": loss,
        "loss_l1": loss_l1,
        "loss_l2": loss_l2,
        "psnr": psnr,
        "predictions": {
        },
        "eval_metric_ops": {
            "mean-psnr": tf.metrics.mean(psnr),
            "psnr": tf.metrics.mean(psnr),
        }
    }


def _default_hparams(args):
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
        learning_rate=args.lr
    )
    return hparams


def main(argv):
    """The main function to train model"""

    del argv
    for i in range(1, args.stage + 1):
        steps = i * args.mini_steps
        args.lr *= args.lr_decay
        hparams = _default_hparams(args)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " steps:", steps, " lr=", args.lr)
        utils1.train_and_eval(
            model_dir=args.model_dir,
            model_fn=model_fn,
            create_input_fn=create_input_fn,
            hparams=hparams,
            steps=steps,
            batch_size=args.batch_size,
            sync_replicas=args.sync_replicas,
            save_checkpoints_steps=args.save_checkpoints_steps,
            save_summary_steps=args.save_summary_steps,
        )


if __name__ == "__main__":
    tf.app.run()
