# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
BASE=sys.argv[1] # must have the last slash!
def one_file_loss_compute(num):
    def int_shape(x):
        return list(map(int, x.get_shape()))

    def compute_log_prob_x(z, sum_log_det_jacobians):

        # y is assumed to be in standard normal distribution
        # 1/sqrt(2*pi)*exp(-0.5*x^2)
        z=tf.reshape(z,[12,32,32,3])
        zs = int_shape(z)
        K = zs[1] * zs[2] * zs[3]  # dimension of the Gaussian distribution

        log_density_z = -0.5 * tf.reduce_sum(tf.square(z), [1, 2, 3]) - 0.5 * K * np.log(2 * np.pi)

        log_density_x = log_density_z + sum_log_det_jacobians

        # to go from density to probability, one can
        # multiply the density by the width of the
        # discrete probability area, which is 1/256.0, per dimension.
        # The calculation is performed in the log space.
        log_prob_x = log_density_x - K * tf.log(256.0)

        return log_prob_x

    def loss(z, sum_log_det_jacobians):
      m= compute_log_prob_x(z, sum_log_det_jacobians)
      return -tf.reduce_sum(m)

    gen_par2=[]
    jacs2=[]
    with open(BASE+"cifar_test"+str(num)+"_output_0.txt", "r") as f:
        content = f.read()
        lines = [line.split(" ") for line in content.split("\n")]
        lst = []
        for line in lines:
            changeline=[]
            for ele in line:
                if(ele!=''):
                    tmp = float(ele)
                    changeline.append(tmp)
            lst += changeline
        # print(lst)
        gen_par2=lst

    with open(BASE+"cifar_test"+str(num)+"_output_1.txt", "r") as f:
        content = f.read()
        lines = [line.split(" ") for line in content.split("\n")]
        lst = []
        for line in lines:
            changeline=[]
            for ele in line:
                if(ele!=''):
                    tmp = float(ele)
                    changeline.append(tmp)
            lst += changeline
        # print(lst)
        jacs2=lst
    # file_prefix="D:\\OGNet\\real-nvp-new\\res_bin\\cifar_test0_output_1.bin"
    # datasett = np.fromfile(file_prefix, dtype=np.float32)
    # jacs2=datasett


    def list_split(items, n):
        return [items[i:i+n] for i in range(0, len(items), n)]



    # gen_par2_one = list_split(gen_par2, 3072)
    # jacs2_one=list_split(jacs2,1)
    test_losses=[]
    # # 0:gen 1:jac
    # for i in range(12):

    loss_gen_test=[]
    with tf.Session() as sess:
        r=sess.run(loss(gen_par2, jacs2))
        m = r / (2 * np.log(2.) * np.prod((32, 32, 3)) * 12)
        loss_gen_test.append(m)
    #print(loss_gen_test)
    #print((np.prod((32,32,3))))
    bits_per_dim_test=loss_gen_test[0]
    test_losses.append(bits_per_dim_test) # `test_losses` is the loss per sample

    test_loss_gen = np.mean(test_losses)
    return test_loss_gen

loss=[]
for i in range(0,832):
    one_loss=one_file_loss_compute(i) # `loss` is the loss per batch
    #print(one_loss)
    loss.append(one_loss)

print(np.mean(loss)) # total loss

# 9.9781449812955411
##3.6483, test bits_per_dim = 3.6627