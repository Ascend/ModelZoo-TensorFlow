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


from time import time
import tensorflow as tf
from npu_bridge.npu_init import *
from config import get_config
from RunModel import RunModel
from absl import flags
import os
from datasets.common import read_images_from_tfrecords
from benchmark.eval_util import compute_errors
import numpy as np

eval_data_dir = '/mnt/cloud_disk/zhy/hmr_datasets/mpi_inf_3dhp/test'
flags.DEFINE_string('eval_data_dir', eval_data_dir, 'where to load eval data')

def eval(config, model, sess):
    all_tfrecords = sorted([
        os.path.join(config.eval_data_dir, path) for path in os.listdir(config.eval_data_dir)])
    
    raw_errors, raw_errors_pa = [], []
    for record in all_tfrecords:
        images, kps, gt3ds = read_images_from_tfrecords(record, config.img_size, sess)
        results = run_model(images, config, model)
        # Evaluate!
        # Joints 3D are in COCOplus format now. First 14 are H36M joints.
        pred3ds = results['joints3d'][:, :14, :]
        # Convert to mm!
        errors, errors_pa = compute_errors(gt3ds * 1000., pred3ds * 1000.)
        raw_errors.append(errors)
        raw_errors_pa.append(errors_pa)
    
    MPJPE = np.mean(np.hstack(raw_errors))
    PA_MPJPE = np.mean(np.hstack(raw_errors_pa))

    return MPJPE, PA_MPJPE

def run_model(images, config, model):
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
            joints, verts, cams, joints3d, thetas = model.predict(
                images_wdummy, get_theta=True)
            joints = joints[:num_here]
            verts = verts[:num_here]
            cams = cams[:num_here]
            joints3d = joints3d[:num_here]
            thetas = thetas[:num_here]
        else:
            joints, verts, cams, joints3d, thetas = model.predict(
                images_here, get_theta=True)

        all_joints.append(joints)
        all_verts.append(verts)
        all_cams.append(cams)
        all_joints3d.append(joints3d)
        all_thetas.append(thetas)

    preds = {
        'verts': np.vstack(all_verts),
        'cams': np.vstack(all_cams),
        'joints': np.vstack(all_joints),
        'joints3d': np.vstack(all_joints3d),
        'thetas': np.vstack(all_thetas)
    }

    return preds


if __name__ == '__main__':
    config = get_config()
    if not config.load_path:
        raise Exception('Must specify a model to use to predict!')
    if 'model.ckpt' not in config.load_path:
        raise Exception('Must specify a model checkpoint!')
    # define sess config
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # close remap
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    model = RunModel(config=config, sess=sess)
    MPJPE, PA_MPJPE = eval(config, model, sess)
    sess.close()
    print('Metrics: MPJPE= %.1f  PA_MPJPE= %.1f' % (MPJPE, PA_MPJPE))