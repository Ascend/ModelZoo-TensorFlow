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
import cv2 as cv
import skimage.color as sc
from skimage.measure import compare_psnr as psnr

parser = argparse.ArgumentParser(description="RFN")
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--lr_dir_test", type=str, default=None)
parser.add_argument("--hr_dir_test", type=str, default=None)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--sync_replicas", type=int, default=-1)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--path", type=str, default=None)


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def create_test_input_fn():
    """Create input function"""

    if(not os.path.exists(args.lr_dir_test) or
       not os.path.exists(args.hr_dir_test)):
        raise IOError("Testing dataset not found")

    def input_fn():
        """The input function"""

        def parser(lr_imgs, hr_imgs):
            """Read image from the file system"""

            lr_imgs = tf.image.decode_png(tf.read_file(lr_imgs), channels=3)
            hr_imgs = tf.image.decode_png(tf.read_file(hr_imgs), channels=3)

            lr_imgs = tf.div(tf.to_float(lr_imgs), 255)
            hr_imgs = tf.div(tf.to_float(hr_imgs), 255)

            return lr_imgs, hr_imgs

        hr_imgs = sorted(os.listdir(args.hr_dir_test))
        lr_imgs = sorted(os.listdir(args.lr_dir_test))
        hr_imgs = [os.path.join(args.hr_dir_test, img) for img in hr_imgs]
        lr_imgs = [os.path.join(args.lr_dir_test, img) for img in lr_imgs]
        img_pair = (lr_imgs, hr_imgs)

        dataset = tf.data.Dataset.from_tensor_slices(img_pair)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.repeat(1).batch(1)
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
    loss = loss_l1 / 2 + loss_l2 / 2

    return {
        "loss": loss,
        "loss_l1": loss_l1,
        "loss_l2": loss_l2,
        "predictions": {
            "predict": sr_img,
            "labels": labels,
        },
    }


def eval(
        model_dir,
        steps,
        model_fn,
        input_fn,
        hparams,
        keep_checkpoint_every_n_hours=0.5,
        save_checkpoints_secs=180,
        save_summary_steps=50,
        sync_replicas=0,
        path=None):
    """Evaluate the model with estimator"""

    run_config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        save_checkpoints_secs=save_checkpoints_secs,
        save_summary_steps=save_summary_steps)

    estimator = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=utils1.standard_model_fn(
            model_fn,
            steps,
            run_config,
            sync_replicas=sync_replicas),
        params=hparams,
        config=run_config)

    if path != None:
        print("Model path is ", path)
        output = estimator.predict(input_fn=input_fn(), checkpoint_path=path)
    else:
        print("Loading Default Path")
        output = estimator.predict(input_fn=input_fn())
    return output


def quantize(img):
    """Quantize the numpy array to range (0, 255)"""
    return img.clip(0, 255).round().astype(np.uint8)


def test(args):
    """Test the model with test dataset"""

    hparams = _default_hparams()
    output = eval(
        model_dir=args.model_dir,
        model_fn=model_fn,
        input_fn=create_test_input_fn,
        hparams=hparams,
        steps=args.steps,
        save_checkpoints_secs=600,
        sync_replicas=args.sync_replicas,
        path=args.path
    )

    record = []
    scale = args.scale
    for pred_no, pred_dict in enumerate(output):
        sr = pred_dict["predict"][:, :, [2, 0, 1]] / 255
        hr = pred_dict["labels"][:, :, [2, 0, 1]] / 255
        hr = quantize(sc.rgb2ycbcr(hr)[:, :, 0])
        sr = quantize(sc.rgb2ycbcr(sr)[:, :, 0])
        hr = hr[scale:-scale, scale:-scale, ...]
        sr = sr[scale:-scale, scale:-scale, ...]
        record.append(psnr(hr, sr, data_range=255))

    print("Mean PSNR of test dataset is %.10f" % np.mean(record))


def _default_hparams():
    """Returns default or overridden user-specified hyperparameters."""
    
    hparams = tf.contrib.training.HParams(
        learning_rate=args.lr
    )
    return hparams


if __name__ == "__main__":
    test(args)
