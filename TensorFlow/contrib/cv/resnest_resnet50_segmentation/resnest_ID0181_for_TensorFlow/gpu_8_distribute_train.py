# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from resnest import deeplabv3_resnest50

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "model_save_dir", "./save_gpu_model/resnest.ckpt",
    "The config json file corresponding to the pre-trained resnest model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_data", "./train_data/cityscapes_train.tfrecords",
    "The config json file corresponding to the pre-trained resnest model. "
    "This specifies the model architecture.")

flags.DEFINE_bool(
    "is_training", True,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "num_class", 19,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "img_N", 2975,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "img_H", 512,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "img_W", 512,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "batch_size", 12,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "epochs", 500,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_float(
    "learning_rate", 0.00001,
    "The config json file corresponding to the pre-trained resnest model.")


def _parse_read(tfrecord_file):
    features = {
        'image':
            tf.io.FixedLenFeature((), tf.string),
        "label":
            tf.io.FixedLenFeature((), tf.string),
        'height':
            tf.io.FixedLenFeature((), tf.int64),
        'width':
            tf.io.FixedLenFeature((), tf.int64),
        'channels':
            tf.io.FixedLenFeature((), tf.int64)
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(
        image, [parsed['height'], parsed['width'], parsed['channels']])
    label = tf.decode_raw(parsed['label'], tf.uint8)
    label = tf.reshape(label, [parsed['height'], parsed['width'], 1])
    
    combined = tf.concat([image, label], axis=-1)
    combined = tf.random_crop(combined, (flags.img_H, flags.img_W, 4))
    
    
    
    image = combined[:, :, 0:3]
    label = combined[:, :, 3:4]
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int64)
    return image, label[:, :, 0:]


def _augmentation(image, label):
    
    print("----------------------", image, label)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 32)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_saturation(image, 0.5, 1.5)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, 0.5, 1.5)
    return image, label[:, :, 0]


def _preprocess(image, label):
    image = image / 255.
    image = image - [0.406, 0.456, 0.485]
    image = image/[0.225, 0.224, 0.229]
    return image, label


def readdata(tf_data, batch_size, true_or_false):
    dataset = tf.data.TFRecordDataset(tf_data)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    if true_or_false:
        dataset = dataset.map(_augmentation)
        dataset = dataset.map(_preprocess)
        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.repeat(flags.epochs)
    else:
        dataset = dataset.repeat(flags.epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def train_step(dist_inputs):
    def focal_loss(logits, label, mask, gamma=2):
        epsilon = 1.e-10
        label = tf.one_hot(label, flags.num_class)
        probs = tf.nn.softmax(logits)
        probs = tf.clip_by_value(probs, epsilon, 1.)
        gamma_weight = tf.multiply(label, tf.pow(
            tf.subtract(1., probs), gamma))  
        loss = -label*tf.log(probs)*gamma_weight
        print("-----------------------------", loss)
        loss = tf.reduce_sum(loss, axis=-1)
        print("---------------loss--------------", loss)
        print("---------------mask--------------", mask)
        mask = tf.cast(mask, tf.float32)
        loss = loss * mask
        print("-----------------------------", loss)
        loss = tf.reduce_mean(loss)
        return loss

    def step_fn(inputs):
        image, labels = inputs
        fake = tf.fill(
            [int(flags.batch_size/2), flags.img_H, flags.img_W], 255)
        fake = tf.cast(fake, tf.int64)
        one = tf.ones([int(flags.batch_size/2),
                       flags.img_H, flags.img_W], tf.int64)
        zeros = tf.zeros(
            [int(flags.batch_size/2), flags.img_H, flags.img_W], tf.int64)
        print("--------------------fake------------------", fake)
        print("--------------------labels-----------------", labels)
        mask = tf.where(tf.equal(labels, fake), zeros, one)
        labels = tf.where(tf.equal(labels, fake), zeros, labels)
        out = deeplabv3_resnest50(image, flags.is_training, [
                                  flags.img_H, flags.img_W])
        loss = focal_loss(out, labels, mask)
        
        train_op = optimizer.minimize(loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.control_dependencies([train_op]):
                return tf.identity(loss)
  

    per_replica_losses = mirrored_strategy.experimental_run_v2(
        step_fn, args=(dist_inputs,))
  
    mean_loss = mirrored_strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    return mean_loss


mirrored_strategy = tf.distribute.MirroredStrategy()
print("创建成功！！")
with mirrored_strategy.scope():
    optimizer = tf.train.AdamOptimizer(flags.learning_rate)
    print("模型定义成功！")
    dataset = readdata(flags.train_data, flags.batch_size, True)
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    print(dist_dataset.__dict__['_cloned_datasets'])
    print("数据分发成功")
    input_iterator = dist_dataset.make_initializable_iterator()
    iterator_init = input_iterator.initialize()
    var_init = tf.global_variables_initializer()

    loss = train_step(input_iterator.get_next())

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([var_init, iterator_init])
    saver = tf.train.Saver(max_to_keep=10)
    
    for epoch in range(flags.epochs):
        for step in range(int(flags.img_N/flags.batch_size)):
            tra_loss = sess.run(loss)
            
            print('Epoch %d, step %d,train loss = %.4f' %
                  (epoch + 1, step + 1,  tra_loss))
        saver.save(sess, flags.model_save_dir, global_step=epoch)
