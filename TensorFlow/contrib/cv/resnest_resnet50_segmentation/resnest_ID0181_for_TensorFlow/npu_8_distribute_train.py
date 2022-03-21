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
import os
import numpy as np
from resnest import deeplabv3_resnest50
import time

from npu_bridge.npu_init import *
from npu_bridge.estimator import *
from npu_bridge.estimator.npu import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
from npu_bridge.npu_init import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#from hccl.split.api import set_split_strategy_by_idx

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "model_save_dir", "./save_npu_model/resnest.ckpt",
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
    "batch_size", 10,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_integer(
    "epochs", 1,
    "The config json file corresponding to the pre-trained resnest model. ")

flags.DEFINE_float(
    "learning_rate", 0.00001,
    "The config json file corresponding to the pre-trained resnest model.")

flags.DEFINE_string(
    "pre_model", "./save_gpu_model/resnest.ckpt-46",
    "The config json file corresponding to the pre-trained resnest model. "
    "This specifies the model architecture.")

train_tf = FLAGS.train_data
epochs = FLAGS.epochs
batch_size = FLAGS.batch_size
img_H = FLAGS.img_H
img_W = FLAGS.img_W
img_N = FLAGS.img_N
is_training = True
num_class = FLAGS.num_class
pre_model = FLAGS.pre_model
print("----------------------------",pre_model)
model_save_dir = FLAGS.model_save_dir
rank_size = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID'))
os.environ["GE_USE_STATIC_MEMORY"] = "1"


def read_data(tf_file, batch_size, is_training):
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
        image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width'], 1])
        
        combined = tf.concat([image, label], axis=-1)
        combined = tf.random_crop(combined, (img_H, img_W, 4))
        image = combined[:, :, 0:3]
        label = combined[:, :, 3:4]
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        return image, label[:, :, 0:]

    def _augmentation(image, label):
        
        print("----------------------", image, label)
        with tf.Session().as_default():
            if tf.random.uniform(()).eval() > 0.5:
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(label)
            
            if tf.random.uniform(()).eval() > 0.5:
                image = tf.image.random_brightness(image,32)
            
            if tf.random.uniform(()).eval() > 0.5:
                image = tf.image.random_saturation(image, 0.5, 1.5)
            
            if tf.random.uniform(()).eval() > 0.5:
                image = tf.image.random_contrast(image, 0.5, 1.5)
        return image, label[:, :, 0]

    def _preprocess(image, label):
        image = image / 255.
        image = image - [0.406, 0.456, 0.485]
        image = image/[0.225, 0.224, 0.229]
        return image, label

    dataset = tf.data.TFRecordDataset(tf_file)
    rank_size = int(os.getenv('RANK_SIZE'))  
    rank_id = int(os.getenv('RANK_ID'))  
    dataset = dataset.shard(rank_size, rank_id)  
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(_parse_read, num_parallel_calls=16) 
    if is_training:
        dataset = dataset.map(_augmentation, num_parallel_calls=16)
        dataset = dataset.map(_preprocess, num_parallel_calls=16)
        dataset = dataset.repeat()
    else:
        dataset = dataset.map(_preprocess, num_parallel_calls=16)
        dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=batch_size*10)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    
    return images_batch, labels_batch


def training_op(log, label, mask):
    def focal_loss(logits, label, mask, gamma=2):
        epsilon = 1.e-10
        label = tf.one_hot(label, 19)
        probs = tf.nn.softmax(logits)
        probs = tf.clip_by_value(probs, epsilon, 1.)
        gamma_weight = tf.multiply(label, tf.pow(tf.subtract(1., probs), gamma))  
        loss = -label*tf.log(probs)*gamma_weight
        print("-----------------------------", loss)
        loss = tf.reduce_sum(loss, axis=-1)
        print("-----------------------------", loss)
        loss = loss * mask
        print("-----------------------------", loss)
        loss = tf.reduce_mean(loss)
        return loss

    loss = focal_loss(log, label, mask)
    optimizer = tf.train.AdamOptimizer(0.0001)
    optimizer = NPUDistributedOptimizer(optimizer)
    
    loss_scaling = 2 ** 25
    grads = optimizer.compute_gradients(loss * loss_scaling)
    for i, (grad, var) in enumerate(grads):
        if grad is not None:
            grads[i] = (grad/loss_scaling, var)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads)
        return train_op, loss


def evaluating_op(log, label, num_classes):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU


def main(_):
    images_batch, labels_batch = read_data(train_tf, batch_size, is_training)
    print("---------------images----------------", images_batch)
    print("-----------------label--------------", labels_batch)

    inputx = tf.placeholder(
        tf.float32, shape=[batch_size, img_H, img_W, 3], name="inputx")
    inputy = tf.placeholder(
        tf.int64, shape=[batch_size, img_H, img_W],  name="inputy")
    inputm = tf.placeholder(
        tf.float32, shape=[batch_size, img_H, img_W],  name="inputm")

    out = deeplabv3_resnest50(inputx, is_training, [img_H, img_W])
    train_op, train_loss = training_op(out, inputy, inputm)

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    
    
    
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    #custom_op.parameter_map["enable_data_pre_proc"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    #custom_op.parameter_map["hcom_parallel"].b = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    #set_split_strategy_by_idx([260,335])
    saver = tf.train.Saver(max_to_keep=10)
    if pre_model != None:
        print(pre_model)
        saver.restore(sess,pre_model)
    print("Training start....")
    try:
        for epoch in range(epochs):
            print("epoch",epoch)
            s = time.time()
            for step in range(int(img_N/batch_size-260)): #add
                st = time.time()
                x_in, y_in = sess.run([images_batch, labels_batch])
                m_in = np.where(y_in == 255, 0, 1)
                y_in = np.where(y_in == 255, 0, y_in)
                _, tra_out, tra_loss = sess.run([train_op, out, train_loss], feed_dict={inputx: x_in, inputy: y_in, inputm: m_in})
                p_acc, m_acc, miou, _ = evaluating_op(tra_out, y_in, num_class)
                if (step+1) % 2== 0: #add
                    print('time %.4f,rank_id %d,Epoch %d, step %d, train loss = %.4f, pixel accuracy= %.2f, mean accuracy= %.2f, miou = %.2f' %(time.time() - st, rank_id, epoch+1, step+1, tra_loss, p_acc, m_acc, miou)) #add
                    #print('time %.4f,rank_id %d,Epoch %d, step %d, train loss = %.4f' %(time.time() - st, rank_id, epoch+1, step+1, tra_loss)) #add

            #if rank_id == 0:
            #    saver.save(sess, model_save_dir, global_step=epoch)
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        print("Training Done")


if __name__ == '__main__':
    tf.app.run()
