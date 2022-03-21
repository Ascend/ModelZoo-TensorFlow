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
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import my_horse2zebra
import numpy as np
# tfds.disable_progress_bar()
# AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset, metadata = tfds.load('cycle_gan/horse2zebra',
#                               with_info=True, as_supervised=True)

# train_horses, train_zebras = dataset['trainA'], dataset['trainB']
# test_horses, test_zebras = dataset['testA'], dataset['testB']

(train_horses,train_zebras),(test_horses,test_zebras) = my_horse2zebra.load_horse2zebra("./datasets/horse2zebra/",get_new=False,detype=np.int16)
# 鍏堣浆鎴愭湁绗﹀彿鐨 涓嶈兘杞垚int8 鍥犱负int8 蹇呯劧鏄細鎴柇棣栦綅 鍋氱鍙蜂綅鐨
print(train_horses.shape)
plt.imshow(train_horses[0, :, :,:])
plt.show()
a = train_horses[0, :, :]
plt.hist(a.flatten(), bins=80, color='c')
plt.xlabel("Pix 0 ~ 255")
plt.ylabel("Frequency")
plt.show()
a = a/127.5-1.0
print(a.dtype)
plt.imshow(a) # 瀹冨彧鑳借瘑鍒0~255鐨勬暣鍨 0~1鐨勬诞鐐瑰瀷 涓ょ 濡傛灉鏄诞鐐瑰瀷 鑷姩鎴彇0~1.0鍖洪棿 濡傛灉鏄暣褰 鎴彇0~255鍖洪棿 鎵浠ュ綊涓鍖栧悗鏄剧ず寮傚父鏃舵甯哥幇璞
plt.show()
plt.hist(a.flatten(), bins=80, color='c')
plt.xlabel("Pix -1 ~ 1")
plt.ylabel("Frequency")
plt.show()

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image
# 灏嗗浘鍍忓綊涓鍖栧埌鍖洪棿 [-1, 1] 鍐呫
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
def random_jitter(image):
    # 璋冩暣澶у皬涓 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method="nearest")
                            # 鏈杩戦偦鎻掑 鑾峰緱鍥剧墖鐨勭缉鏀炬瘮gama_x=缂╂斁鍓峹/缂╂斁鍚巟 gama_y=缂╂斁鍓峺/缂╂斁鍚巠 
                            # 瀵逛簬鎻掑煎悗鍥惧儚鐨 x浣嶇疆鍍忕礌 = x*gama_x浣嶇疆鐨勫師鍥惧儚鍍忕礌 鍥涜垗浜斿叆鍙栨渶杩戠殑浣嶇疆鍗冲彲 
                            # 鍥惧儚鎻掑(涓鑸槸鏀惧ぇ)涔嬫垜瑙  鏈変簡缂╂斁姣斾箣鍚 1*缂╂斁姣 褰㈡垚灏忎簬1鐨勯棿闅 浠ヨ繖涓棿闅 褰㈡垚娴偣鍧愭爣 鍗冲甫鍏ヨ绠楃殑宸寚鐐 璁＄畻瀹屾墍鏈夋诞鐐瑰潗鏍囧悗  鍗宠绠楀畬浜嗘暣涓彃鍊煎悗鐨勬暣鏁板潗鏍囦綅缃殑鍍忕礌鍊 
                            # 娴偣鍧愭爣鏄湪鍘熷鍥惧儚鍧愭爣绯讳腑鐨 鑰屾诞鐐瑰潗鏍囦竴涓瀵瑰簲鎻掑煎悗鐨勫浘鍍忕殑鍍忕礌鐐
                            # 鎵浠 鏈杩戦偦鎻掑 娴偣鍧愭爣鐨勫儚绱犳槸鍙栦笌鍏舵渶杩戠殑鍘熷鍥惧儚鍧愭爣鐨勫儚绱犲 
                            #      鍙岀嚎鎬ф彃鍊 娴偣鍧愭爣鐨勫儚绱犳槸鍙栦笌鍏舵渶杩戠殑鍥涗釜鍘熷鍥惧儚鍧愭爣绾挎ц绠楄屾潵 鍗冲厛妯悜绾挎ф彃鍊 鍐嶇旱鍚戠嚎鎬ф彃鍊
                            # 杩欑娴偣鍧愭爣鐨勬蹇垫湁鍒╀簬鐞嗚В鎻掑艰繃绋
                            # 濡傛灉鏄缉灏 鍒欑缉鏀炬瘮澶т簬1 鍒欎篃绫讳技鐨
    # 闅忔満瑁佸壀鍒 256 x 256 x 3
    image = random_crop(image)
    # 闅忔満闀滃儚
    image = tf.image.random_flip_left_right(image)

    return image
def preprocess_image_train(image, label=None):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    image = normalize(image)
    return image

train_horses = tf.map_fn(preprocess_image_train,train_horses.astype(np.float32))
print(train_horses)
plt.imshow(train_horses[0, :, :,:])
plt.show()
a = train_horses.numpy() #numpy 鏄痶ensor鐨勪竴涓柟娉  璋冪敤numpy() 鏂规硶 杩斿洖涓涓猲umpy鐭╅樀

plt.hist(a[0,:,:,:].flatten(), bins=80, color='c')
plt.xlabel("Pix 0 ~ 255")
plt.ylabel("Frequency")
plt.show()
train_zebras = tf.map_fn(preprocess_image_train,train_zebras.astype(np.float32))
test_horses = tf.map_fn(preprocess_image_train,test_horses.astype(np.float32))
test_zebras = tf.map_fn(preprocess_image_train,test_zebras.astype(np.float32))

train_horses=tf.data.Dataset.from_tensor_slices(train_horses).shuffle(BUFFER_SIZE).batch(1)
train_zebras=tf.data.Dataset.from_tensor_slices(train_zebras).shuffle(BUFFER_SIZE).batch(1)
test_horses=tf.data.Dataset.from_tensor_slices(test_horses).shuffle(BUFFER_SIZE).batch(1)
test_zebras=tf.data.Dataset.from_tensor_slices(test_zebras).shuffle(BUFFER_SIZE).batch(1)

"""
tf.data.Dataset.from_tensor_slices 鏋勫缓鐨勬槸涓涓嚜宸辩殑瀹炰緥 鏃㈠彲浠ist杞负鍒楄〃 姣忎釜鍏冪礌閮芥槸涓涓猙atch 
涔熷彲浠ョ敤 iter() 鏋勫缓杩唬鍣
"""
print("**************************************")
sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))
plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
plt.show()

plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
plt.show()

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_zebra = generator_g(sample_horse)
to_horse = generator_f(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

plt.show()

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5
def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 濡傛灉瀛樺湪妫鏌ョ偣锛屾仮澶嶆渶鏂扮増鏈鏌ョ偣
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 40
def generate_images(model, test_input):
    prediction = model(test_input)
        
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # 鑾峰彇鑼冨洿鍦 [0, 1] 涔嬮棿鐨勫儚绱犲间互缁樺埗瀹冦
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def train_step(real_x, real_y):
    # persistent 璁剧疆涓 Ture锛屽洜涓 GradientTape 琚娆″簲鐢ㄤ簬璁＄畻姊害銆
    with tf.GradientTape(persistent=True) as tape:
        # 鐢熸垚鍣 G 杞崲 X -> Y銆
        # 鐢熸垚鍣 F 杞崲 Y -> X銆
        
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x 鍜 same_y 鐢ㄤ簬涓鑷存ф崯澶便
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # 璁＄畻鎹熷け銆
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        
        # 鎬荤敓鎴愬櫒鎹熷け = 瀵规姉鎬ф崯澶 + 寰幆鎹熷け銆
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
    # 璁＄畻鐢熸垚鍣ㄥ拰鍒ゅ埆鍣ㄦ崯澶便
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)
    
    # 灏嗘搴﹀簲鐢ㄤ簬浼樺寲鍣ㄣ
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                generator_f.trainable_variables))
    
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))
    
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))

                    
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n+=1

    clear_output(wait=True)
    # 浣跨敤涓鑷寸殑鍥惧儚锛坰ample_horse锛夛紝浠ヤ究妯″瀷鐨勮繘搴︽竻鏅板彲瑙併
    generate_images(generator_g, sample_horse)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


# 鍦ㄦ祴璇曟暟鎹泦涓婅繍琛岃缁冪殑妯″瀷銆
for inp in test_horses.take(5):
    generate_images(generator_g, inp)