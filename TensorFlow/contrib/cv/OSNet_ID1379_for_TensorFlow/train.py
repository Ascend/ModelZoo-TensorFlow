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

import os
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.general import categorical_crossentropy_label_smoothing
from tensorflow.keras.optimizers import SGD
from data import DataGenerator
from osnet import OSNet
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse
from npu_bridge.npu_init import *
# Set training parameters
image_shape = (128, 64, 3)  # h x w x c
use_label_smoothing = True


def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training OSNet.')
    parser.add_argument('--initial_lr', help='initial learning rate', type=float, default=0.065)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--num_epoch', help='total epoch to train', type=int, default=100)
    parser.add_argument('--train_image_dir', help='path to train_image', type=str, default='/home/dingwei/osnet/dataset/Market-1501-v15.09.15/bounding_box_train')
    # parser.add_argument('--output_path', help='path to output', type=str, default='/home/dingwei/osnet/osnet_tf/output')
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args=None):
    # Npu setting
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.parameter_map["use_off_line"].b = True  # 必须显式开启，在昇腾AI处理器执行训练
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    K.set_session(sess)

    args = parse_args(args)
    # Preprocess data
    train_image_filenames = sorted([filename for filename in os.listdir(args.train_image_dir) if filename.endswith(".jpg")])
    train_image_paths = [os.path.join(args.train_image_dir, name) for name in train_image_filenames]
    train_person_ids = [name[:4] for name in train_image_filenames]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_person_ids)
    train_person_ids_encoded = label_encoder.transform(train_person_ids)
    num_person_ids = len(set(train_person_ids_encoded))

    train_img_paths, val_img_paths, train_person_ids, val_person_ids = train_test_split(
        train_image_paths, train_person_ids_encoded, test_size=0.1, random_state=2021,
        stratify=train_person_ids_encoded)
    print(
        f"# train images: {len(train_img_paths)}, # val images: {len(val_img_paths)}, # image labels: {num_person_ids}")

    # Contruct and show model
    baseline_model = OSNet(751).model
    print(baseline_model.summary())

    loss = categorical_crossentropy_label_smoothing if use_label_smoothing else "categorical_crossentropy"
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = SGD(lr=args.initial_lr, momentum=0.9)
    baseline_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Set lr decay
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的9/10
        if epoch % 30 == 0 and epoch != 0:
            lr = K.get_value(baseline_model.optimizer.lr)
            K.set_value(baseline_model.optimizer.lr, lr * 0.9)
            print("lr changed to {}".format(lr * 0.9))
        return K.get_value(baseline_model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)

    train_generator = DataGenerator(train_img_paths, train_person_ids, batch_size=args.batch_size,
                                    num_classes=num_person_ids, shuffle=True, augment=True)
    val_generator = DataGenerator(val_img_paths, val_person_ids, batch_size=args.batch_size, num_classes=num_person_ids)

    # Train model
    baseline_model.fit(
        train_generator,
        epochs=args.num_epoch,
        # validation_data=val_generator,
        callbacks=[reduce_lr],
        shuffle=True,
    )

    # Save model
    baseline_model.save_weights("osnet.h5")
    print("Training completed and model saved.")
    sess.close()

if __name__ == "__main__":
    main()
