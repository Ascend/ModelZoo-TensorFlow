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
# Copyright 2022 Huawei Technologies Co., Ltd
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


import npu_device as npu
npu.open().as_default()

import tensorflow as tf
import shutil
import os
import argparse


from models.ultrafast import UltraFastNet
from utils import losses, metrics
from utils.datasets import llamas_dataset, labelme_dataset


parser = argparse.ArgumentParser(description='Train Ultrafast Net')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-images', type=int, default=None)
parser.add_argument('--prefetch-size', type=int, default=256)
parser.add_argument('--resnet-weights', default=None)
parser.add_argument('--base-path', default='')
parser.add_argument('--llamas-path', '--data-path', default='./data/llamas/')
parser.add_argument('--labelme-path', default=[], action='append')
parser.add_argument('--output-path', default='./')
parser.add_argument('--model-name', default='ultrafast')

args = parser.parse_args()
print(args)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PREFETCH_SIZE = args.prefetch_size

NUM_IMAGES = args.max_images
VAL_SPLIT = 0.3

NUM_LANES = 2
CLS_SHAPE = (100, 20, NUM_LANES)
IMAGE_SHAPE = (288, 800, 3)

RESNET_WEIGHTS = args.resnet_weights

BASE_PATH = args.base_path
LLAMAS_PATH = os.path.join(BASE_PATH, args.llamas_path)
LABELME_PATHS = [os.path.join(BASE_PATH, path) for path in args.labelme_path]

OUTPUT_PATH = args.output_path
LOG_DIR = os.path.join(OUTPUT_PATH, 'logs_ultrafast')
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, 'checkpoints_ultrafast')
MODEL_DIR = os.path.join(OUTPUT_PATH, 'model')

MODEL_NAME = '{}.tf'.format(args.model_name)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

print('Preparing datasets')
llamas_train_ds = llamas_dataset(os.path.join(LLAMAS_PATH, 'labels', 'train', '*', '*.json'), CLS_SHAPE, IMAGE_SHAPE)
llamas_valid_ds = llamas_dataset(os.path.join(LLAMAS_PATH, 'labels', 'valid', '*', '*.json'), CLS_SHAPE, IMAGE_SHAPE)

labelme_ds = None
for path in LABELME_PATHS:
    ds = llamas_dataset(os.path.join(LLAMAS_PATH, '*.json'), CLS_SHAPE, IMAGE_SHAPE)
    if labelme_ds is None:
        labelme_ds = ds
    else:
        labelme_ds = labelme_ds.concatenate(ds)
        
train_ds = llamas_train_ds
if labelme_ds is not None:
    train_ds = train_ds.concatenate(labelme_ds)

valid_ds = llamas_valid_ds

train_ds = train_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)

print('Preparing model')
model = UltraFastNet(num_lanes=NUM_LANES, 
                     size=IMAGE_SHAPE[0:2],
                     cls_dim=CLS_SHAPE, 
                     use_aux=False, 
                     resnet_weights=RESNET_WEIGHTS)

adam = tf.keras.optimizers.Adam()

print('compiling model')
model.compile(optimizer=adam, loss=losses.ultrafast_loss, metrics=[ metrics.ultrafast_accuracy])
print('model compiled')
model.summary()


shutil.rmtree(LOG_DIR, ignore_errors=True)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'ckpt')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_ultrafast_accuracy',
    mode='max',
    save_best_only=True)

print('Training model')
history = model.fit(train_ds, 
                    epochs=EPOCHS,
                    validation_data=valid_ds,
                    callbacks=[#tensorboard_callback,
                                model_checkpoint_callback],
                    verbose=1)
model.optimizer = None
model.compiled_loss = None
model.compiled_metrics = None
model.save(MODEL_PATH, save_format='tf')

