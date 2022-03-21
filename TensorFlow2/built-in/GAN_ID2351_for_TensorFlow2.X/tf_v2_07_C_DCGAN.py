#
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
#from __future__ import absolute_import, division, print_function, unicode_literals
"""
鍩轰簬tensorflow 楂橀樁API
CGAN  Conditional GAN 
鎹熷け鍑芥暟: 鍩轰簬SGAN鐨凩oss 鍒ゅ埆鍣ㄨ緭鍑轰负姒傜巼鍊奸渶瑕乻igmoid
缃戠粶缁撴瀯: 澶氬眰鍗风Н缁撴瀯 鍒ゅ埆鍣ㄥ湪鍗风Н灞傚悗鐨勫叏杩炴帴灞俢oncat One_Hot鏉′欢  鑰 鐢熸垚鍣ㄥ湪寮澶碿oncat One_Hot 鏉′欢
鏁版嵁褰㈠紡: 甯﹀嵎绉眰 鏁版嵁鏄犲皠鍒-1 1 鍖洪棿
鐢熸垚鍣: tanh 鏄犲皠鍒-1 1 涔嬮棿 杩庡悎鏁版嵁鏍煎紡
鍒ゅ埆鍣: sigmoid 鏄犲皠鍒0 1 涔嬮棿 杩庡悎loss鍏紡鐨勭害鏉
鍒濆鍖: xavier鍒濆鍖  鍗宠冭檻杈撳叆杈撳嚭缁村害鐨 glorot uniform
璁粌锛 鍒ゅ埆鍣ㄥ拰鐢熸垚鍣ㄥ悓鏃惰缁 鍚屾璁粌 涓嶅亸閲嶄换涓鏂
"""
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import my_layers
from tensorflow.keras import layers
import time
(train_images,train_labels),(_, _) = my_mnist.load_data(get_new=False,
                                                        normalization=False,
                                                        one_hot=True,
                                                        detype=np.float32)
train_images = (train_images.astype('float32')-127.5)/127.5
train_labels = (train_labels.astype('float32')-0.5)/0.5                                                    
train_images = train_images.reshape(train_images.shape[0], 28, 28,1)
print(train_labels[0])
plt.imshow(train_images[0, :, :,0], cmap='gray')
plt.show()

    

class Discriminator(tf.keras.Model):
    def __init__(self,in_shape,label_dim):
        super(Discriminator,self).__init__()
        """
        鍙嶅嵎绉拰dense灞傞噰鐢ㄥ亸缃 鍚勮嚜2鍙傛暟 
        2+2+2+2+2=10 涓鍏卞崄涓弬鏁颁釜鏁(鎸囩嫭绔嬪ぇ鍙傛暟self.w self.b鐨勪釜鏁)
        """
        self.Conv2d_1 = my_layers.Conv2D(input_shape=in_shape,out_depth=64,filter_size=[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
        self.LeakyReLU_1 = my_layers.LeakyReLU(in_shape=self.Conv2d_1.out_shape)
        self.DropOut_1 = my_layers.Dropout(in_shape=self.LeakyReLU_1.out_shape,dropout_rate=0.3)
        
        self.Conv2d_2 = my_layers.Conv2D(input_shape=self.DropOut_1.out_shape,out_depth=128,filter_size=[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
        self.LeakyReLU_2 = my_layers.LeakyReLU(in_shape=self.Conv2d_2.out_shape)
        self.DropOut_2= my_layers.Dropout(in_shape=self.LeakyReLU_2.out_shape,dropout_rate=0.3)
        next_shape = 1
        for i in self.DropOut_2.out_shape:
            next_shape *= i 
        # self.Dense = my_layers.Dense(next_shape,units=1)
        self.Dense = my_layers.Dense(next_shape,units=100)
        self.Dense_conditional_1 = my_layers.Dense(self.Dense.out_dim+label_dim,units=50)
        self.Dense_conditional_2 = my_layers.Dense(self.Dense_conditional_1.out_dim,units=1)
    @tf.function
    def call(self,x,y,training=True):
        conv2_l1 = self.Conv2d_1(x)
        leakey_relu_l1 = self.LeakyReLU_1(conv2_l1,training)
        dropout_l1 = self.DropOut_1(leakey_relu_l1,training)
        conv2_l2 = self.Conv2d_2(dropout_l1)
        leakey_relu_l2 = self.LeakyReLU_2(conv2_l2,training)
        dropout_l2 = self.DropOut_2(leakey_relu_l2,training)
        dense_l3 =  self.Dense(tf.reshape(dropout_l2,[dropout_l2.shape[0],-1]),training)
        # l3_out = tf.nn.sigmoid(dense_l3) 瑕佸拰鏍囩缁熶竴缁村害 涓嶈兘鍐嶆槸sigmoid 鑰屽簲璇ユ槸tanh
        l3_out = tf.nn.tanh(dense_l3)
        l4_out = tf.nn.leaky_relu(self.Dense_conditional_1(tf.concat([l3_out,y],axis=1),training))
        l5_out = tf.nn.sigmoid(self.Dense_conditional_2(l4_out,training))
        return l5_out

d = Discriminator(in_shape=[28,28,1],label_dim=10)
x = train_images[0:128, :, :,:]
y = train_labels[0:128,:]
print(d(x,y,training=False))#琛屽悜閲忕粺涓杈撳叆  鑰宐atch鏄鍚戦噺鍦ㄥ垪鏂瑰悜鍫嗗彔鍚庣殑鐭╅樀 
# print(d(x,training=True))#琛屽悜閲忕粺涓杈撳叆  鑰宐atch鏄鍚戦噺鍦ㄥ垪鏂瑰悜鍫嗗彔鍚庣殑鐭╅樀 
print(len(d.trainable_variables))

class Generator(tf.keras.Model):
    def __init__(self,in_dim):
        super(Generator,self).__init__()
        """
        bn灞備袱涓弬鏁 
        鍙嶅嵎绉拰dense灞備笉閲囩敤鍋忕疆 鍚勮嚜鍙湁涓涓弬鏁 
        1+2+1+2+1+2+1=10 涓鍏卞崄涓弬鏁颁釜鏁(鎸囩嫭绔嬪ぇ鍙傛暟self.w self.b鐨勪釜鏁)
        """
        self.Dense_1 = my_layers.Dense(in_dim,7*7*256,use_bias=False)
        self.BacthNormalization_1 = my_layers.BatchNormalization(in_shape=self.Dense_1.out_dim)
        self.LeakyReLU_1 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_1.out_shape)
        
        self.Conv2dTranspose_2 = my_layers.Conv2DTranspose(in_shape=[7,7,256],out_depth=128,kernel_size=[5,5],strides=[1,1],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_2.out_shape == [7,7,128]
        self.BacthNormalization_2 = my_layers.BatchNormalization(in_shape=self.Conv2dTranspose_2.out_shape)
        self.LeakyReLU_2 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_2.out_shape)

        self.Conv2dTranspose_3 = my_layers.Conv2DTranspose(in_shape=self.LeakyReLU_2.out_shape,out_depth=64,kernel_size=[5,5],strides=[2,2],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_3.out_shape == [14,14,64]
        self.BacthNormalization_3 = my_layers.BatchNormalization(in_shape=self.Conv2dTranspose_3.out_shape)
        self.LeakyReLU_3 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_3.out_shape)

        self.Conv2dTranspose_4 = my_layers.Conv2DTranspose(in_shape=self.LeakyReLU_3.out_shape,out_depth=1,kernel_size=[5,5],strides=[2,2],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_4.out_shape == [28,28,1]
    @tf.function
    def call(self,x,y,training=True):
        x = tf.concat([x,y],axis=1) #1缁村害鐩稿姞 鍏朵粬缁村害鍥哄畾涓嶅彉
        dense_l1 = self.Dense_1(x,training)
        #tf.print(dense_l1)
        bn_l1 = self.BacthNormalization_1(dense_l1,training)
        #tf.print(bn_l1) batch_normalization 鍦ㄨ缁冩椂 濡傛灉batch sizez鏄1 鍒欎細鐩存帴褰掗浂 鍥犱负浼氬噺鍘诲潎鍊
        lr_l1 = self.LeakyReLU_1(bn_l1,training)
        #tf.print(lr_l1)
        conv2d_tr_l2 = self.Conv2dTranspose_2(tf.reshape(lr_l1,[-1,7,7,256]))
        bn_l2 = self.BacthNormalization_2(conv2d_tr_l2,training)
        lr_l2 = self.LeakyReLU_2(bn_l2,training)

        conv2d_tr_l3 = self.Conv2dTranspose_3(lr_l2)
        bn_l3 = self.BacthNormalization_3(conv2d_tr_l3,training)
        lr_l3 = self.LeakyReLU_3(bn_l3,training)

        conv2d_tr_l4 = self.Conv2dTranspose_4(lr_l3)
        l4_out = tf.nn.tanh(conv2d_tr_l4)
        return l4_out

g = Generator(100+10)
z = tf.random.normal((2,100))
y = train_labels[0:2,:]
image = g(z,y,training=False)
for i in range(image.shape[0]):
    plt.imshow(tf.reshape(image[i],(28,28)), cmap='gray')
    plt.show()

# image = g(z,training=True)
# for i in range(image.shape[0]):
#     plt.imshow(tf.reshape(image[i],(28,28)), cmap='gray')
#     plt.show()
print(len(g.trainable_variables))
print(d(image,y,training=False))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)#鍒ゅ埆鍣ㄥ凡缁弒igmoid 鎵浠ユ槸false
def d_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def g_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
BATCH_SIZE = 128
z_dim = 100
num_examples_to_generate = 100
# seed = tf.random.normal([num_examples_to_generate, z_dim],mean=0.0,stddev=1.0)
seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)
num_list = []
for i in range(10):
    num_list += [i]*10
print(num_list)
seed_lable= tf.one_hot(num_list,depth=10,on_value=1.0,off_value=-1.0,axis=-1,dtype=tf.float32) #axis鐞嗚В鎴愭垜浠姞鍏ョ殑娣卞害10 鍦ㄦ渶缁堢粨鏋滀腑鐨勮酱搴忓彿
print(seed_lable)
seed = [seed,seed_lable]



@tf.function
def train_step(images,labels):
    # z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(z,labels,training=True)
        real_output = d(images,labels,training=True)
        fake_output = d(generated_images,labels,training=True)
        gen_loss = g_loss(fake_output)
        disc_loss = d_loss(real_output,fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))

def train(train_images,train_labels,epochs):
    index = list(range(train_images.shape[0]))
    np.random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
    for epoch in range(epochs):
        start = time.time()
        while True:
            try:
                x_real_bacth = next(images_batches)
                y_label_bacth = next(labels_batches)
                train_step(x_real_bacth,y_label_bacth)
            except StopIteration:
                del images_batches
                del labels_batches
                np.random.shuffle(index)
                train_images = train_images[index]
                train_labels = train_labels[index]
                images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
                labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
                break
        generate_and_save_images(g,epoch + 1,seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
def generate_and_save_images(model,epoch,test_input):
    predictions = model(test_input[0],test_input[1],training=False)
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*127.5+127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('./DCGAN_C/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

print(time)
train(train_images,train_labels,EPOCHS)