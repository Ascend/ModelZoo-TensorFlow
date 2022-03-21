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
import logging
import sys
import time
from losses import disc_loss, gen_en_loss
from misc import get_fixed_random, generate_images
import npu_device as npu

def train(config, gen, disc_f, disc_h, disc_j, model_en, train_data):
    # Start training
    # Define optimizers
    disc_optimizer = tf.optimizers.Adam(learning_rate=config.lr_disc,
                                        beta_1=config.beta_1_disc,
                                        beta_2=config.beta_2_disc)

    gen_en_optimizer = tf.optimizers.Adam(learning_rate=config.lr_gen_en,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)

    # Define Logging to Tensorboard
    logdir=f'{config.result_path}/{config.model}_{config.dataset}_{time.strftime("%Y-%m-%d--%H-%M-%S")}'
    summary_writer = tf.summary.create_file_writer(logdir)

    fixed_z, fixed_c = get_fixed_random(config, num_to_generate=100)  # fixed_noise is just used for visualization.

    # Define metric
    metric_loss_gen_en = tf.keras.metrics.Mean()
    metric_loss_disc = tf.keras.metrics.Mean()

    # Start training
    epoch_tf = tf.Variable(0, trainable=False, dtype=tf.float32)
    it = iter(train_data)
    for epoch in range(config.num_epochs):
        logging.info(f'Start epoch {epoch+1} ...')  # logs a message.
        print("Start epoch {} ...".format(epoch+1))  # logs a message.
        epoch_tf.assign(epoch)
        start_time = time.time()
        tf.summary.trace_on(graph=True, profiler=False) 
        train_epoch(it, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
                    metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config)
        epoch_time = time.time()-start_time

        # Save results
         print("epoch : {}----Disc_loss : {}----Gen_loss : {}----sec/step : {}".format(epoch+1, metric_loss_disc.result(), metric_loss_gen_en.result(), epoch_time))
        logging.info(f'Epoch {epoch+1}: Disc_loss: {metric_loss_disc.result()}, Gen_loss: {metric_loss_gen_en.result()}, Time: {epoch_time}')
        with summary_writer.as_default():
            tf.summary.scalar('Generator and Encoder loss',metric_loss_gen_en.result(),step=epoch)
            tf.summary.scalar('Discriminator loss', metric_loss_disc.result(),step=epoch)

        metric_loss_gen_en.reset_states()

        metric_loss_disc.reset_states()
        # Generated images and reconstructed images
        gen_image  = generate_images(gen, fixed_z, fixed_c, config)
        with summary_writer.as_default():
            tf.summary.image('Generated Images', tf.expand_dims(gen_image,axis=0),step=epoch)
        with summary_writer.as_default():
            tf.summary.trace_export(
                name="autograph",
                step=0)

@tf.function
def loop_train(it, steps, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer,
                metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, conditional, D_G_ratio):
    for i in tf.range(steps):
        image, label = next(it)
        if not conditional:
            label = None
        train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
                metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, D_G_ratio)

def train_epoch(it, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer,
                metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config):
    # remaining_steps = [i for i,_ in enumerate(train_data)][-1] + 1 # 剩余Steps数
    remaining_steps = 234
    # 设置循环次数
    base_loop_size = 234  # 基准npu loop size
    while remaining_steps >= base_loop_size:
        npu.set_npu_loop_size(base_loop_size)  # 设置循环下沉次数
        loop_train(it, tf.constant(base_loop_size), gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
                   metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config.conditional, config.D_G_ratio)
        remaining_steps -= base_loop_size
    if remaining_steps > 0:
        npu.set_npu_loop_size(remaining_steps)  # 设置循环下沉次数
        loop_train(it, tf.constant(remaining_steps), gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
            metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config) 

def train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
               metric_loss_gen_en, batch_size, cont_dim, D_G_ratio):
    for _ in range(D_G_ratio):
        fake_noise = tf.random.truncated_normal([batch_size, cont_dim])
        with tf.GradientTape(persistent=True) as gen_en_tape, tf.GradientTape() as en_tape:
            fake_img = gen(fake_noise, label, training=True)
            latent_code_real = model_en(image, training=True)
            with tf.GradientTape(persistent=True) as disc_tape:
                real_f_to_j, real_f_score = disc_f(image, label, training=True)
                fake_f_to_j, fake_f_score = disc_f(fake_img, label, training=True)
                real_h_to_j, real_h_score = disc_h(latent_code_real, training=True)
                fake_h_to_j, fake_h_score = disc_h(fake_noise, training=True)
                real_j_score = disc_j(real_f_to_j, real_h_to_j, training=True)
                fake_j_score = disc_j(fake_f_to_j, fake_h_to_j, training=True)
                d_loss = disc_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
                g_e_loss = gen_en_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
        grad_disc = disc_tape.gradient(d_loss, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables)
        disc_optimizer.apply_gradients(zip(grad_disc, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables))
        metric_loss_disc.update_state(d_loss)  # upgrade the value in metrics for single step.

    grad_gen_en = gen_en_tape.gradient(g_e_loss, gen.trainable_variables + model_en.trainable_variables)
    gen_en_optimizer.apply_gradients(zip(grad_gen_en, gen.trainable_variables + model_en.trainable_variables))
    metric_loss_gen_en.update_state(g_e_loss)

    del gen_en_tape, en_tape
    del disc_tape




