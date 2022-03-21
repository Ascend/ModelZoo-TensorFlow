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
WGAN-GP  Wasserstein GAN 鐨勮缁冩敼杩
鎹熷け鍑芥暟: WGAN W 璺濈loss  鑰冨療鐪熷亣鏍锋湰鐨勫垎甯冨樊寮 鍒ゅ埆鍣ㄩ渶瑕佹弧瓒矻ipschitz-1 Lipschitz-K 绾︽潫 鎵浠ヤ笉鑳藉姞sigmoid绾︽潫鑼冨洿
        閫氳繃瀵硅緭鍏ョ殑姹傚 寰楀埌鍏充簬杈撳叆鐨勫鏁 鍗虫潈鍊糤鐨勫嚱鏁 浣滀负姝ｅ垯椤  瀵瑰叾鍊艰繘琛岀洿鎺ョ害鏉 浠庤 婊¤冻Lipschitz-1 Lipschitz-K 绾︽潫
缃戠粶缁撴瀯: MLP 鑷冲皯鏈変袱灞 鍗宠緭鍏ュ眰鍚 鑷冲皯1涓腑闂村眰 鐒跺悗鏄緭鍑哄眰 甯哥敤128鑺傜偣 
鏁版嵁褰㈠紡: 涓嶅甫鍗风Н 娌℃湁娣卞害缁  鍥剧墖鍘嬬缉鍒0 1 涔嬮棿 
鐢熸垚鍣: sigmoid 鏄犲皠鍒0 1 涔嬮棿 杩庡悎鏁版嵁鏍煎紡
鍒ゅ埆鍣: 鏈鍚庝竴灞 娌℃湁sigmoid 娌℃湁relu 鐩存帴鏄痬atmul杩愮畻缁撴灉  杩庡悎loss鍏紡鐨勭害鏉 鍦ㄥ叏鍩熷唴瀵绘壘婊¤冻Lipschitz-1 Lipschitz-K 绾︽潫鐨勫嚱鏁
鍒濆鍖: xavier鍒濆鍖  鍗宠冭檻杈撳叆杈撳嚭缁村害鐨 glorot uniform
璁粌锛 鍒ゅ埆鍣5娆 鐢熸垚鍣1娆
"""
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
(train_images,train_labels),(_, _) = my_mnist.load_data(get_new=False,
                                                        normalization=True,
                                                        one_hot=True,
                                                        detype=np.float32)
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
print(train_labels[0])
plt.imshow(train_images[0, :, :], cmap='gray')
plt.show()

class Dense(layers.Layer):
    def __init__(self, input_dim, units):
        super(Dense,self).__init__()
        # initializer = tf.initializers.glorot_uniform()
        initializer = tf.initializers.glorot_normal()
        self.w = tf.Variable(initial_value=initializer(shape=(input_dim,units),dtype=tf.float32),trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(1,units),dtype=tf.float32),trainable=True)#鑺傜偣鐨勫亸缃篃鏄鍚戦噺 鎵嶅彲浠ユ甯歌绠 鍗冲鍫嗗彔鐨刡atch 閮芥槸鍔犺浇鍗曚釜batch鍐
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.matmul(x,self.w)+self.b
            return y
        else:
            y = tf.matmul(x,self.w)+self.b
            return y 
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1 = Dense(28*28,128)
        self.dense2= Dense(128,32)
        self.dense3 = Dense(32,1)
    @tf.function
    def call(self,x,training=True):
        """
        batch*dim+batch*10 鍦╥ndex_1缁村害缁勫悎 鍏朵綑缁村害涓嶅彉
        """
        x = tf.reshape(x,[-1,784]) #reshape 涓嶆敼鍙樺師濮嬬殑鍏冪礌椤哄簭 杩欏緢閲嶈 闃叉鍙樺舰鏃跺彉鎴愯浆缃 蹇界暐batch澶у皬  鍙叧娉ㄥ悗闈㈢殑缁村害涓鑷
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3_out = tf.nn.sigmoid(self.dense3(l2_out,training))
            return l3_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            # l3_out = tf.nn.sigmoid(self.dense3(l2_out,training))
            l3_out = self.dense3(l2_out,training)
            return l3_out
    @tf.function       
    def to_clip(self):
        for weight in self.trainable_variables:
            weight.assign(tf.clip_by_value(weight,clip_value_min=-0.01,clip_value_max=0.01))
d = Discriminator()
x = train_images[0:2, :, :]
print(d(x,training=False))#琛屽悜閲忕粺涓杈撳叆  鑰宐atch鏄鍚戦噺鍦ㄥ垪鏂瑰悜鍫嗗彔鍚庣殑鐭╅樀 
print(len(d.trainable_variables))

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1 = Dense(100,32)
        self.dense2 = Dense(32,128)
        self.dense3 = Dense(128,784)
    @tf.function
    def call(self,x,training=True):
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training))
        l3_out = tf.reshape(l3,[-1,28,28])
        return l3_out

g = Generator()
z = tf.random.normal((1,100))
image = g(z,training=False)
plt.imshow(tf.reshape(image,(28,28)), cmap='gray')
plt.show()

print(d(image,training=False))

def d_loss(real_output, fake_output):
    total_loss = -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)#鐢╞atch 鍧囧奸艰繎鏈熸湜 鐒跺悗渚濇嵁鍏紡 max  鎵浠ュ彇鍙  -E(real)+E(fake)  鍋歮in
    return total_loss
def g_loss(fake_output):
    return -tf.reduce_mean(fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0,beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0,beta_2=0.9)

EPOCHS = 400
BATCH_SIZE = 128
z_dim = 100
num_examples_to_generate = 100
seed = tf.random.normal([num_examples_to_generate, z_dim],mean=0.0,stddev=1.0)
# seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)

@tf.function
def D_train_step(images,labels):
    z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as disc_tape:
        generated_images = g(z,training=True)
        e = tf.random.uniform((images.shape[0],1,1),0.0,1.0)
        mid_images = e*images+(1-e)*generated_images
        with tf.GradientTape() as gradient_penalty:
            gradient_penalty.watch(mid_images)
            inner_loss = d(mid_images,training=True)
        penalty = gradient_penalty.gradient(inner_loss,mid_images)
        penalty_norm = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1)# 杩欐槸鎴戣嚜宸辫涓虹殑  鍥犱负鍙湁姊害澶т簬1鐨勬墠闇瑕佷紭鍖栧搰
        # penalty_norm = 10.0*tf.math.square(tf.norm(penalty,ord='euclidean')-1) 杩欐槸鎸夌収绠楁硶鎰挎剰
        real_output = d(images,training=True)
        fake_output = d(generated_images,training=True)
        # disc_loss = d_loss(real_output,fake_output)
        disc_loss = d_loss(real_output,fake_output)+tf.reduce_mean(penalty_norm)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))

@tf.function
def G_train_step(images,labels):
    z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape:
        generated_images = g(z,training=True)
        fake_output = d(generated_images ,training=True)
        gen_loss = g_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))

def train(train_images,train_labels,epochs):
    break_flag = 0
    index = list(range(train_images.shape[0]))
    np.random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
    for epoch in range(epochs):
        start = time.time()
        while True:
            for i in range(5):
                try:
                    x_real_bacth = next(images_batches)
                    y_label_bacth = next(labels_batches)
                    D_train_step(x_real_bacth,y_label_bacth)
                except StopIteration:
                    del images_batches
                    del labels_batches
                    np.random.shuffle(index)
                    train_images = train_images[index]
                    train_labels = train_labels[index]
                    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
                    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
                    break_flag = 1
                    break
            if break_flag == 0: # 鍒ゅ埆鍣ㄨ缁5娆 鐒跺悗杩涜涓娆＄敓鎴愬櫒
                G_train_step(x_real_bacth,y_label_bacth)
            else:
                break_flag = 0
                break
        generate_and_save_images(g,epoch + 1,seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input,training=False)
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*255.0, cmap='gray')
        plt.axis('off')
    plt.savefig('./WGAN_GP/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

print(time)
train(train_images,train_labels,EPOCHS)