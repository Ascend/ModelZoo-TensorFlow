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
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image
import numpy as np
from tensorflow.python.framework.ops import convert_to_tensor
TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor =  1 - 0.75/2


class vaegan(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, model_path, data_ob, latent_dim, sample_path, learnrate_init):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.repeat_num = repeat
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.latent_dim = latent_dim
        self.sample_path = sample_path
        self.learn_rate_init = learnrate_init
        self.log_vars = []

        self.channel = 3
        self.output_size = data_ob.image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp = tf.random_normal(shape=[self.batch_size, self.latent_dim])

        self.dataset = tf.data.Dataset.from_tensor_slices(
            convert_to_tensor(self.data_ob.train_data_list, dtype=tf.string))
        self.dataset = self.dataset.map(lambda filename : tuple(tf.py_func(self._read_by_function,
                                                                            [filename], [tf.double])), num_parallel_calls=16)
        self.dataset = self.dataset.repeat(self.repeat_num)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.next_x = tf.squeeze(self.iterator.get_next())
        self.training_init_op = self.iterator.make_initializer(self.dataset)

    def build_model_vaegan(self):

        self.z_mean, self.z_sigm = self.Encode(self.images)
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        self.x_tilde = self.generate(self.z_x, reuse=False)
        self.l_x_tilde, self.De_pro_tilde = self.discriminate(self.x_tilde)

        self.x_p = self.generate(self.zp, reuse=True)

        self.l_x,  self.D_pro_logits = self.discriminate(self.images, True)
        _, self.G_pro_logits = self.discriminate(self.x_p, True)

        #KL loss
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)

        # D loss
        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_pro_logits) - d_scale_factor, logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        # G loss
        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits) - g_scale_factor, logits=self.G_pro_logits))
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde) - g_scale_factor, logits=self.De_pro_tilde))

        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_tilde_loss

        # preceptual loss(feature loss)
        self.LL_loss = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.l_x_tilde, self.l_x), [1,2,3]))

        #For encode
        self.encode_loss = self.kl_loss/(self.latent_dim*self.batch_size) - self.LL_loss / (4 * 4 * 256)

        #for Gen
        self.G_loss = self.G_fake_loss + self.G_tilde_loss - 1e-6*self.LL_loss

        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("LL_loss", self.LL_loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)


    # save images
    def dump(self):

        # get real images and pb out images
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Initialzie the iterator
            sess.run(self.training_init_op)

            sess.run(init)
            self.saver.restore(sess, self.saved_model_path)

            step = 0

            while step <= 2000:

                next_x_images = sess.run(self.next_x)

                real_images, sample_images = sess.run([self.images, self.x_tilde], feed_dict={self.images: next_x_images})
                # shape is (64,64,64,3)
                save_images(sample_images[0:self.batch_size], [self.batch_size/8, 8], '{}/test_{}_pb.png'.format(self.sample_path, step))
                save_images(real_images[0:self.batch_size], [self.batch_size/8, 8], '{}/test_{}_real.png'.format(self.sample_path, step))
                next_x_images.tofile("{}/npu_input_{}.bin".format(self.sample_path, step))
                step = step + 1
            print("dump END")



    # save acl images
    def save_acl_image(self):

        step = 0

        while step <= 2000:
            # (64,64,64,3)
            pred = np.fromfile("{}/xacl_out_bin/xacl_out_{}_output_00_000.bin".format("./Data", step)
                               , dtype="float32").reshape(64, 64, 64, 3)
            save_images(pred[0:self.batch_size],[self.batch_size / 8, 8],
                        "{}/test_{}_pred.png".format(self.sample_path, step))
            step = step + 1

    # get psnr
    def compare_mean(self):

        def read_img(path):
            return tf.image.decode_image(tf.read_file(path))

        def psnr(tf_img1, tf_img2):
            return tf.image.psnr(tf_img1, tf_img2, max_val=255)

        def main(path1, path2):
            t1 = read_img(path1)
            t2 = read_img(path2)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                y = sess.run(psnr(t1, t2))
                print(y)
            return y

        step = 0
        mean2 = 0

        # compare 100 times
        while step <= 100:
            p1 = "{}/test_{}_real.png".format(self.sample_path, step)
            p3 = "{}/test_{}_pred.png".format(self.sample_path, step)

            mean2 = mean2 + main(p1, p3)
            step = step + 1



        mean2 = mean2/(step - 1)
        print("compare END")
        print("psnr:", mean2)



    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
            conv2= tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= tf.nn.relu(batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4= tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv4= tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl , output_size=1, scope='dis_fully2')

            return middle_conv, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
            d2 = tf.reshape(d1, [self.batch_size, 8, 8, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 16, 16, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 32, 32, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 64, 64, 32], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
            d5 = de_conv(d4, output_shape=[self.batch_size, 64, 64, 3], name='gen_deconv5', d_h=1, d_w=1)

            return tf.nn.tanh(d5)

    def Encode(self, x):

        with tf.variable_scope('encode') as scope:

            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 8 * 8])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1 , output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54 , 108, 108], channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                           is_grayscale=False)
        real_images = np.array(array)
        return real_images











