
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

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from tqdm import tqdm
import threading
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset, DatasetFetcher
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from core.cocoeval import COCOevaluator
import time
import npu_device

flags.DEFINE_string('model', '', 'yolov4, yolov3')
flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
flags.DEFINE_string('anno_converted', '', 'path to converted coco val annotation')
flags.DEFINE_string('gt_anno_path', '', 'path to origin coco val annotation')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('batch_size', 48, 'batch size per device')
flags.DEFINE_string('precision_mode', 'allow_fp32_to_fp16', 'optional: "allow_fp32_to_fp16","allow_mix_precision"')
flags.DEFINE_integer('stage1_epoch', 15, 'stage 1 freezes output conv layer')
flags.DEFINE_integer('stage2_epoch', 23, 'stage 2 unfreezes output conv layer')
flags.DEFINE_boolean('over_dump', False, 'overflow dump')
flags.DEFINE_string('over_dump_path', '', 'path to dump overflow data')
flags.DEFINE_boolean('data_dump_flag', False, 'data dump')
flags.DEFINE_string('data_dump_path', '', 'path to dump data')
flags.DEFINE_integer('data_dump_step', 10, 'num of dump steps')
flags.DEFINE_boolean('profiling', False, 'profiling dump')
flags.DEFINE_string('data_path', '', 'path of converted train annotation file')
#flags.DEFINE_integer('loss_scale', 1, 'loss scale, dynamic if set to 0')
flags.DEFINE_integer('train_worker_num', 8, 'train worker num')
flags.DEFINE_boolean('eval_only', False, 'skip train process')
flags.DEFINE_boolean('mosaic', True, 'activate mosaic data augmentation')
flags.DEFINE_integer('rank', 0, 'rank of current device')
flags.DEFINE_integer('rank_size', 1, 'rank size of current device')
flags.DEFINE_integer('perf', 0, 'run steps for perf')
tic = 0
e_tic = 0

def main(_argv):
    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = 'all'
       
    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = 'all'
    npu_device.global_options().precision_mode = FLAGS.precision_mode
#    npu_device.global_options().fusion_switch_file = ''
    npu_device.global_options().modify_mixlist = '../../../ops_info.json'
    npu_device.global_options().hcom_parallel=True
    npu_device.open().as_default()
    
    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./output"
    isfreeze = False
    steps_per_epoch = len(trainset)
    print('steps_per_epoch: ', steps_per_epoch)
    first_stage_epochs = FLAGS.stage1_epoch
    second_stage_epochs = FLAGS.stage2_epoch
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if not FLAGS.weights:
        print("Training from scratch")
    else:
        print('Restoring weights ... ')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    npu_device.distribute.broadcast(model.trainable_variables, 0)


    # define training step function
    @tf.function
    def train_execute(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = ciou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        npu_device.distribute.all_reduce(gradients, 'mean')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return ciou_loss, conf_loss, prob_loss, total_loss
            
    def train_step(image_data, target):
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * ((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
        optimizer.lr.assign(lr.numpy())
        ciou_loss, conf_loss, prob_loss, total_loss = train_execute(image_data, target)
        # if FLAGS.rank == 0:
        print("=> STEP %4d/%4d   lr: %.6f   ciou_loss: %4.2f   conf_loss: %4.2f   prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(), ciou_loss, conf_loss, prob_loss, total_loss), end='', flush=True)
        global_steps.assign_add(1)

    if not FLAGS.eval_only:
        mutex_data_iter = threading.Lock()
        mutex_sess_run = threading.Lock()
        global e_tic
        e_tic = time.time()
        for epoch in range(first_stage_epochs + second_stage_epochs):
            if epoch < first_stage_epochs:
                if not isfreeze:
                    isfreeze = True
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
            elif epoch >= first_stage_epochs:
                if isfreeze:
                    isfreeze = False
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        unfreeze_all(freeze)
            trainset.rewind()
            def worker():
                global tic
                tic = time.time()
                fetcher = DatasetFetcher(trainset)
                while True:
                    with mutex_data_iter:
                        try:
                            annotations = next(trainset)
                        except StopIteration:
                            break
                    image_data, target, _, _, _, _ = fetcher.process_annotations(annotations)
                    with mutex_sess_run:
                        # if FLAGS.rank == 0:
                        rstart = time.time()
                        train_step(image_data, target)
                        # if FLAGS.rank == 0:
                        duration = time.time() - tic
                        print(' ,global_step/sec: %.2f ,duration: %.2f'%((1 / duration), duration), flush=True)
                        tic = time.time()
                        if FLAGS.perf and (FLAGS.perf < global_steps.numpy()):
                            break
            threads = []
            for i in range(FLAGS.train_worker_num):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join()
            
            # if FLAGS.rank == 0:
            print('epoch_duration: %d'%(time.time() - e_tic), flush=True)
            e_tic = time.time()
            # print('saving checkpoints', flush=True)
            # checkpoint.save(checkpoint_dir+'/model.ckpt')
    if not FLAGS.perf and FLAGS.rank == 0:
        evaluator = COCOevaluator(model, testset, cfg.TRAIN.INPUT_SIZE, NUM_CLASS, FLAGS)
        evaluator.evaluate()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass