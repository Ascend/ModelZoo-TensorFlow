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
import numpy as np
from PIL import Image
from utils import truncated_noise_sample, restore_img
import datetime
import os
import argparse
import math
import shutil
import imageio
import cv2


def consecutive_category_morphing(arg, img_path, session, fake_img_morphing_op, z_op, y_op, y_end_op, alpha_op,
                                  class1=0, class2=1, fps=2):
    if os.path.exists(img_path):
        shutil.rmtree(img_path)  # delete previous images
    os.makedirs(img_path)

    Z = truncated_noise_sample(arg.batch_size, arg.z_dim, arg.truncation)

    count = 0
    img_paths = []
    for Alpha in [i / 10.0 for i in range(10, -1, -1)]:
        Alpha = np.ones([arg.batch_size, 1]) * Alpha
        fake = session.run(fake_img_morphing_op, feed_dict={z_op: Z, y_op: class1 * np.ones([arg.batch_size]),
                                                            y_end_op: class2 * np.ones([arg.batch_size]),
                                                            alpha_op: Alpha})
        # display a batch of images in a grid
        grid_size = int(arg.batch_size ** 0.5)
        concat_img = np.zeros([grid_size * arg.img_h, grid_size * arg.img_w, 3])
        c = 0
        for i in range(grid_size):
            for j in range(grid_size):
                resized_img = cv2.resize(fake[c], dsize=(arg.img_h, arg.img_w), interpolation=cv2.INTER_LINEAR)
                concat_img[i * arg.img_h: i * arg.img_h + arg.img_h, j * arg.img_w: j * arg.img_w + arg.img_w] = resized_img
                c += 1
        img_path = os.path.join(fake_img_path, "%dto%d_%d.jpg" % (class1, class2, count))
        Image.fromarray(np.uint8(restore_img(concat_img))).save(img_path)
        img_paths.append(img_path)
        count += 1

    # make gif
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    gif_path = os.path.join(fake_img_path, "%dto%d.gif" % (class1, class2))
    imageio.mimsave(gif_path, gif_images, fps=fps)


def generate_img_of_one_class(arg, class_labels, img_name, img_path, session, fake_img_op, z_op, y_op):
    Z = truncated_noise_sample(arg.batch_size, arg.z_dim, arg.truncation)
    fake = session.run(fake_img_op, feed_dict={z_op: Z, y_op: class_labels})

    # display a batch of images in a grid
    grid_size = int(arg.batch_size ** 0.5)
    concat_img = np.zeros([grid_size * arg.img_h, grid_size * arg.img_w, 3])
    c = 0
    for i in range(grid_size):
        for j in range(grid_size):
            resized_img = cv2.resize(fake[c], dsize=(arg.img_h, arg.img_w), interpolation=cv2.INTER_LINEAR)
            concat_img[i * arg.img_h: i * arg.img_h + arg.img_h, j * arg.img_w: j * arg.img_w + arg.img_w] = resized_img
            c += 1
    Image.fromarray(np.uint8(restore_img(concat_img))).save(os.path.join(img_path, img_name))


def generate_img_by_class(arg, img_path, session, fake_img_op, z_op, y_op):
    """For each class, generate some images and display them in a grid"""
    if os.path.exists(img_path):
        shutil.rmtree(img_path)  # delete previous images
    os.makedirs(img_path)

    for nums_c in range(arg.num_classes):
        class_labels = nums_c * np.ones([arg.batch_size])
        img_name = "%d.jpg" % nums_c
        generate_img_of_one_class(arg, class_labels, img_name, img_path, session, fake_img_op, z_op, y_op)


def generate_img(arg, img_path, session, fake_img_op, z_op, y_op):
    """generate fake images with random classes"""
    if os.path.exists(img_path):
        shutil.rmtree(img_path)  # delete previous images
    os.makedirs(img_path)

    for b in range(math.ceil(arg.gen_num // arg.batch_size)):
        Z = truncated_noise_sample(arg.batch_size, arg.z_dim, arg.truncation)
        fake = session.run(fake_img_op, feed_dict={z_op: Z, y_op: np.random.randint(arg.num_classes, size=arg.batch_size)})

        for i in range(arg.batch_size):
            img = cv2.resize(fake[i], dsize=(arg.img_h, arg.img_w), interpolation=cv2.INTER_LINEAR)
            Image.fromarray(np.uint8(restore_img(img))).save(os.path.join(img_path, "%d_fake.jpg" % (b * arg.batch_size + i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # platform arguments (Huawei Ascend)
    parser.add_argument("--chip", type=str, default="gpu", help="run on which chip, cpu or gpu or npu")
    # data arguments
    parser.add_argument("--gen_num", type=int, default=5000, help="number of generated images")
    parser.add_argument("--output", type=str, default=os.path.join("..", "output"), help="output path")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_h", type=int, default=32, help="image height")
    parser.add_argument("--img_w", type=int, default=32, help="image width")
    parser.add_argument("--train_img_size", type=int, default=32,
                        help="image will be resized to this size when training")
    # model arguments
    parser.add_argument("--base_channel", type=int, default=96, help="base channel number for G and D")
    parser.add_argument("--z_dim", type=int, default=120, help="latent space dimensionality")
    parser.add_argument("--truncation", type=float, default=2.0, help="truncation threshold")
    parser.add_argument("--ema", type=bool, default=True, help="use exponential moving average for G")
    parser.add_argument("--shared_dim", type=int, default=128, help="shared embedding dimensionality")
    # function arguments
    parser.add_argument("--function", type=str, default="fake",
                        help="generate fake images or do category morphing (fake / morphing)")
    parser.add_argument("--morphing_class", type=str, default="0_1",
                        help="generate category morphing images between two classes")
    args = parser.parse_args()

    # use different architectures for different image sizes
    if args.train_img_size == 128:
        from networks_128 import Generator, Discriminator
    elif args.train_img_size == 64:
        from networks_64 import Generator, Discriminator
    elif args.train_img_size == 32:
        from networks_32 import Generator, Discriminator

    # get current time
    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    # check output dir
    test_path = os.path.join(args.output, "test")
    fake_img_path = os.path.join(test_path, "fake", str(args.train_img_size))
    image_of_each_class_path = os.path.join(test_path, "image_of_each_class", str(args.train_img_size))
    category_morphing_path = os.path.join(test_path, "category_morphing", str(args.train_img_size))
    # get model path
    model_path = os.path.join(args.output, "model", str(args.train_img_size), "model.ckpt")
    ema_model_path = os.path.join(args.output, "model", str(args.train_img_size), "ema.ckpt")
    resume_path = ema_model_path if args.ema else model_path

    if args.chip == "gpu":
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif args.chip == 'cpu':
        config = tf.ConfigProto()

    train_phase = tf.Variable(tf.constant(False, dtype=tf.bool), name="train_phase")
    # train_phase = tf.placeholder(tf.bool)                           # is training or not
    z = tf.placeholder(tf.float32, [args.batch_size, args.z_dim])   # latent vector
    y = tf.placeholder(tf.int32, [None])                            # class info
    y_end = tf.placeholder(tf.int32, [None])                        # category morphing
    alpha = tf.placeholder(tf.float32, [None, 1])

    G = Generator("generator", args.base_channel)
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        embed_w = tf.get_variable("embed_w", [args.num_classes, args.shared_dim], initializer=tf.orthogonal_initializer())

    if args.function == "fake":
        fake_img = G(z, train_phase, y, embed_w, args.num_classes)
    elif args.function == "morphing":
        fake_img_morphing = G(z, train_phase, y, embed_w, args.num_classes, y_end, alpha)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # load model
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
        saver.restore(sess, resume_path)

        if args.function == "fake":
            # generate fake images
            generate_img(args, fake_img_path, sess, fake_img, z, y)
            # generate fake images for each class
            generate_img_by_class(args, image_of_each_class_path, sess, fake_img, z, y)
        elif args.function == "morphing":
            # category morphing
            classes = args.morphing_class.split("_")
            consecutive_category_morphing(args, category_morphing_path, sess, fake_img_morphing, z, y, y_end, alpha,
                                          class1=int(classes[0]), class2=int(classes[1]), fps=2)
