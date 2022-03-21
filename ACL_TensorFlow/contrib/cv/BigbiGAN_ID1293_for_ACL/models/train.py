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
import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def save_pb(model, output_path='frozen.pb'):
  full_model = tf.function(lambda x1, x2: model(x1, x2))
  full_model = full_model.get_concrete_function(tf.TensorSpec((256, 100), tf.float32),tf.TensorSpec((256), tf.int32))

  # Get frozen ConcreteFunction
  frozen_func = convert_variables_to_constants_v2(full_model)
  frozen_func.graph.as_graph_def()
 
  layers = [op.name for op in frozen_func.graph.get_operations()]
 
  print("-" * 50)
  print("Frozen model inputs: ")
  print(frozen_func.inputs)
  print("Frozen model outputs: ")
  print(frozen_func.outputs)
 
  # Save frozen graph from frozen ConcreteFunction to hard drive
  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=".",
                  name=output_path,
                  as_text=False)


def predict(pb_path, x, y):

  with tf.Graph().as_default() as g:
    output_graph_def = tf.compat.v1.GraphDef()
    init = tf.compat.v1.global_variables_initializer()
    """
    load pb model
    """
    with open(pb_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.graph_util.import_graph_def(output_graph_def) #name是必须的
    
    layers = [op.name for op in g.get_operations()]
  
    """
    enter a text and predict
    """
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        input_z = sess.graph.get_tensor_by_name(
            "import/x1:0")
        input_label = sess.graph.get_tensor_by_name(
            "import/x2:0")
        output = "import/Identity:0"
    
        # you can use this directly
        feed_dict = {
            input_z: x,
            input_label: y
        }
        y_pred_cls = sess.run(output, feed_dict=feed_dict)
    return y_pred_cls
  
def set_up_train(config):
    # Setup tensorflow
    tf.keras.backend.set_learning_phase(0)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # Load dataset
    logging.info('Getting dataset...')
    train_data, _ = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)

    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model_generator = BIGBIGAN_G(config, weight_init)
    
    model_discriminator_f = BIGBIGAN_D_F(config, weight_init)
    model_discriminator_h = BIGBIGAN_D_H(config, weight_init)
    model_discriminator_j = BIGBIGAN_D_J(config, weight_init)
    model_encoder = BIGBIGAN_E(config, weight_init)
    
    # train
    logging.info('Start training...')
    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data)
    
    # Save pb file of generator
    save_pb(model_generator, config.output_pb_path + '/bigbigan.pb')
    # Load bin file
    x1 = np.fromfile(config.input_bin_dir + "/fake_image.bin", dtype='float32')
    x1 = x1.reshape([256,100])
    x2 = np.fromfile(config.input_bin_dir + "/label.bin", dtype='int32')
    x2 = x2.reshape([256])
    
    # Save the results of GPU generated pictures
    gpu_out = model_generator(x1, x2, training=False)
    gpu_out = gpu_out.numpy().astype("float32")
    gpu_out.tofile(config.output_gpu_generated_img + "/gpu_out.bin")
    
    # Finished
    logging.info('Training finished ;)')
