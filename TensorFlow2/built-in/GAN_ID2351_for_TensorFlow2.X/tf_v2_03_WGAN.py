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
WGAN  Wasserstein GAN
鎹熷け鍑芥暟: WGAN W 璺濈loss  鑰冨療鐪熷亣鏍锋湰鐨勫垎甯冨樊寮 鍒ゅ埆鍣ㄩ渶瑕佹弧瓒矻ipschitz-1 Lipschitz-K 绾︽潫 鎵浠ヤ笉鑳藉姞sigmoid绾︽潫鑼冨洿 
        閲囩敤clipping鏂瑰紡绾︽潫
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
import tensorflow.compat.v1 as tf1
import time
import npu_device

FLAGS=tf1.app.flags.FLAGS

tf1.app.flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')
tf1.app.flags.DEFINE_boolean(name='over_dump', default=False,
                    help='if or not over detection, default is False')
tf1.app.flags.DEFINE_boolean(name='data_dump_flag', default=False,
                    help='data dump flag, default is False')
tf1.app.flags.DEFINE_string(name='data_dump_step', default="10",
                    help='data dump step, default is 10')
tf1.app.flags.DEFINE_boolean(name='profiling', default=False,
                    help='if or not profiling for performance debug, default is False') 
tf1.app.flags.DEFINE_string(name='profiling_dump_path', default="/home/data",
                    help='the path to save profiling data')                                      
tf1.app.flags.DEFINE_string(name='over_dump_path', default="/home/data",
                    help='the path to save over dump data')
tf1.app.flags.DEFINE_string(name='data_dump_path', default="/home/data",
                    help='the path to save dump data') 

tf1.app.flags.DEFINE_boolean(name='use_mixlist', default=False,
                    help='whether to enable mixlist, default is True')
tf1.app.flags.DEFINE_boolean(name='fusion_off_flag', default=False,
                    help='whether to enable mixlist, default is True')
tf1.app.flags.DEFINE_string(name='mixlist_file', default="ops_info.json",
                    help='mixlist file name, default is ops_info.json')
tf1.app.flags.DEFINE_string(name='fusion_off_file', default="fusion_switch.cfg",
                    help='fusion_off file name, default is fusion_switch.cfg')

tf1.app.flags.DEFINE_string(name='data_path', default="/home/dataset",
                    help='data path, default is False')
tf1.app.flags.DEFINE_integer(name='train_epochs', default=40,
                    help='train epochs, default is 40')
tf1.app.flags.DEFINE_integer(name='batchSize', default=128,
                    help='Batch Size, default is 128')

def npu_config():
  FLAGS = tf1.app.flags.FLAGS
  npu_config = {}

  if FLAGS.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
    npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if FLAGS.over_dump:
    npu_device.global_options().dump_config.enable_dump_debug = True
    npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
    npu_device.global_options().dump_config.dump_debug_mode = "all"

  if FLAGS.profiling:
    npu_device.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"While_body_while_body_44418_1223/while/model/bert_pretrainer/transformer_encoder/self_attention_mask/mul", \
                        "bp_point":"While_body_while_body_44418_1223/gradient_tape/while/model/bert_pretrainer/transformer_encoder/position_embedding/Pad"}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=FLAGS.precision_mode

  if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
  if FLAGS.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file

  npu_device.open().as_default()

EPOCHS = FLAGS.train_epochs
BATCH_SIZE = FLAGS.batchSize
z_dim = 100
num_examples_to_generate = 100

npu_config()
(train_images,train_labels),(_, _) = my_mnist.load_data(get_new=True,
                                                        normalization=True,
                                                        one_hot=True,
                                                        detype=np.float32,
                                                        data_path=FLAGS.data_path)
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
#print(train_labels[0])
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
            l1_out = tf.nn.relu(self.dense1(x,training=training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training=training))
            l3_out = tf.nn.sigmoid(self.dense3(l2_out,training=training))
            return l3_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training=training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training=training))
            l3_out = self.dense3(l2_out,training=training) #!!!!!! WGAN  鍒ゅ埆鍣ㄤ笉鑳藉姞sigmoid 
            # l3_out = tf.nn.sigmoid(self.dense3(l2_out,training))
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
            l1_out = tf.nn.relu(self.dense1(x,training=training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training=training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training=training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training=training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training=training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training=training))
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

generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=3e-4,epsilon=1e-10)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=3e-4,epsilon=1e-10)

seed = tf.random.normal([num_examples_to_generate, z_dim],mean=0.0,stddev=1.0)
# seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)

@tf.function
def D_train_step(images,labels):
    z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as disc_tape:
        generated_images = g(z,training=True)
        real_output = d(images,training=True)
        fake_output = d(generated_images,training=True)
        disc_loss = d_loss(real_output,fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    d.to_clip()
    return float(disc_loss)

@tf.function
def G_train_step(images,labels):
    z = tf.random.normal([images.shape[0],z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape:
        generated_images = g(z,training=True)
        fake_output = d(generated_images ,training=True)
        gen_loss = g_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    return float(gen_loss)

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
        d_loss = 0
        g_loss=0
        while True:
            for i in range(5):
                try:
                    x_real_bacth = next(images_batches)
                    y_label_bacth = next(labels_batches)
                    d_loss += D_train_step(x_real_bacth,y_label_bacth)
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
                g_loss = G_train_step(x_real_bacth,y_label_bacth)
            else:
                break_flag = 0
                break
        generate_and_save_images(g,epoch + 1,seed)
        if(epoch % 10 ==0):
          print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start),flush='True')
          print ('d_loss: ', np.array(d_loss),flush='True')
          print ('g_loss: ', np.array(g_loss),flush='True')
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input,training=False)
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*255.0, cmap='gray')
        plt.axis('off')
    plt.savefig('./WGAN/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

#print(time)
train(train_images,train_labels,EPOCHS)
