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
from .config import get_config
from absl import flags
import os
from .datasets.common import read_images_from_tfrecords
from .benchmark.eval_util import compute_errors
import numpy as np

from npu_bridge.npu_init import *

test_data_dir = './mpi_inf_3dhp_test_tfrecords'
flags.DEFINE_string('test_data_dir', test_data_dir, 'where to load test data')
pb_path = './pb_model/inference.pb'
flags.DEFINE_string('pb_path', pb_path, 'where to load pb model')

def test(config, sess):
    print('Start testing ...')

    tf.Graph().as_default()
    output_graph_def = tf.GraphDef()
    with open(config.pb_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read()) 
        tf.import_graph_def(output_graph_def, name = "")

    input  = sess.graph.get_tensor_by_name("the_inputs:0")
    output = sess.graph.get_tensor_by_name("the_outputs:0")

    all_tfrecords = sorted([
        os.path.join(config.test_data_dir, path) for path in os.listdir(config.test_data_dir)])
    
    raw_errors, raw_errors_pa = [], []
    for record in all_tfrecords:
        images, gt3ds = get_data(record, sess)
        results = run_model(images, config, sess, input, output)
        # Evaluate!
        # Joints 3D is COCOplus format now. First 14 is H36M joints
        pred3ds = results['joints3d'][:, :14, :]
        # Convert to mm!
        errors, errors_pa = compute_errors(gt3ds * 1000., pred3ds * 1000.)

        raw_errors.append(errors)
        raw_errors_pa.append(errors_pa)
    
    MPJPE = np.mean(np.hstack(raw_errors))
    PA_MPJPE = np.mean(np.hstack(raw_errors_pa))
    print('MPJPE:', MPJPE)
    print('PA_MPJPE:', PA_MPJPE)

def get_data(record, sess):
    images, kps, gt3ds = read_images_from_tfrecords(record, img_size=config.img_size, sess=sess)
    return images, gt3ds

def run_model(images, config, sess, input, output):
    """
    Runs trained model to get predictions on each seq.
    """

    N = len(images)
    all_joints, all_verts, all_cams, all_joints3d, all_thetas = [], [], [], [], []

    # Batch + preprocess..
    batch_size = config.batch_size
    num_total_batches = int(np.ceil(float(N) / batch_size))
    for b in range(num_total_batches):
        start_ind = b * batch_size
        end_ind = (b + 1) * batch_size
        images_here = images[start_ind:end_ind]

        if end_ind > N:
            end_ind = N
            # Need to pad dummy bc batch size is not dynamic,,
            num_here = images_here.shape[0]
            images_wdummy = np.vstack([
                images_here,
                np.zeros((batch_size - num_here, config.img_size,
                          config.img_size, 3))
            ])
            feed_dict = {input: images_wdummy}
            joints3d = sess.run(output, feed_dict=feed_dict)
            joints3d = joints3d[:num_here]
        else:
            feed_dict = {input: images_here}
            joints3d = sess.run(output, feed_dict=feed_dict)

        all_joints3d.append(joints3d)

    preds = {
        'joints3d': np.vstack(all_joints3d)
    }

    return preds


if __name__ == '__main__':
    config = get_config()
    # define sess config
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # close remap
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    test(config, sess)
    sess.close()