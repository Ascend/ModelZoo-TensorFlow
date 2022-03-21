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
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import core.utils as utils
import os
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', "/Volumes/Elements/data/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split()
  for input_value in range(10):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print("calibration image {}".format(fimage[input_value]))
      yield [img_in]
    else:
      continue

def save_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)

  if FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif FLAGS.quantize_mode == 'int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("model saved to: {}".format(FLAGS.output))

def demo():
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  print(output_data)

def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


