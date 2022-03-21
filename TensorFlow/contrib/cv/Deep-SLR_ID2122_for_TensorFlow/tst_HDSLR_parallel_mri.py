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

"""
Created on Wed Sep 12 16:59:44 2018

@author: apramanik


Migration for Huawei ModelArts finished on 21th Nov, 2021

@contributor: Robert LIU
"""

from npu_bridge.npu_init import RewriterConfig, npu_config_proto
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import auxiliaryFunctions as sf
from tqdm import tqdm


def test(data_path, output_path, modelDir):
    """
    This test file is wrapped as a test() function to get direct launched in the training script after training.
    """

    tf.reset_default_graph()

    nImg = 1
    dispind = 0

    # Read the testing data
    test_data = sf.getTestingData(data_path, nImg=nImg)
    org, _, atb, mask, std = test_data['org'], test_data['orgk'], test_data['atb'], test_data['mask'], test_data['minv']
    std = np.expand_dims(std, axis=1)
    std = np.expand_dims(std, axis=2)
    std = np.expand_dims(std, axis=3)
    std = np.tile(std, [1, 12, 256, 232])

    # Load trained model and reconstruct with it
    print('Now loading the model ...')
    rec = np.empty_like(atb)
    tf.reset_default_graph()

    loadChkPoint = tf.train.latest_checkpoint(modelDir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # [ModelArts-NPU] Mixed precision configurations
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    # [ModelArts-NPU] Overflow detection dump configurations
    overflow_dir = os.path.join(output_path, 'overflow')
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(overflow_dir)
    custom_op.parameter_map["enable_dump_debug"].b = True
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        new_saver = tf.train.import_meta_graph(modelDir + '/modelTst.meta')
        new_saver.restore(sess, loadChkPoint)
        graph = tf.get_default_graph()
        maskT = graph.get_tensor_by_name('mask:0')
        fhatT = graph.get_tensor_by_name('fhatT:0')
        atbT = graph.get_tensor_by_name('atb:0')
        wts = sess.run(tf.global_variables())
        for i in tqdm(range(nImg)):
            dataDict = {atbT: atb[[i]], maskT: mask[[i]]}
            rec[i] = sess.run(fhatT, feed_dict=dataDict)

    print('Reconstruction done')

    # Postprocess the data to view results
    org = sf.create_sos(org)
    atb = sf.create_sos(sf.r2c(atb) * std)
    recon = sf.create_sos(sf.r2c(rec) * std)
    error = np.abs(org - recon)
    ssimrec = np.zeros((nImg,), dtype=np.float64)
    psnrRec = np.zeros((nImg,), dtype=np.float64)
    psnrAtb = np.zeros((nImg,), dtype=np.float64)
    for i in range(nImg):
        ssimrec[i] = sf.ssimch(org[i], recon[i])
        psnrAtb[i] = sf.psnr(org[i], atb[i])
        psnrRec[i] = sf.psnr(org[i], recon[i])

    print(' psnrAtb  {0:.3f} psnrRec {1:.3f} ssimrec {2:.3f}  '.format(psnrAtb.mean(), psnrRec.mean(), ssimrec.mean()))

    f = open(os.path.join(output_path, 'test_res.txt'), 'w')
    f.write('{0:.3f} {1:.3f} {2:.3f}'.format(psnrAtb.mean(), psnrRec.mean(), ssimrec.mean()))
    f.close()

    print('********************************')
    recon = recon / recon.max()
    error = error / error.max()
    atb = atb / atb.max()
    org = org / org.max()

    # Display the output images
    plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 0.8))
    plt.clf()
    plt.subplot(141)
    st = 50
    end = 220
    plot(np.abs(org[dispind, st:end, :]))
    plt.axis('off')
    plt.title('Original')
    plt.subplot(142)
    plot(np.abs(atb[dispind, st:end, :]))
    plt.title('Input \n PSNR=' + str(psnrAtb[dispind].round(2)) + ' dB')
    plt.axis('off')
    plt.subplot(143)
    plot(np.abs(recon[dispind, st:end, :]))
    plt.title('H-DSLR, PSNR=' + str(psnrRec[dispind].round(2)) + ' dB')
    plt.axis('off')
    plt.subplot(144)
    plot(error[dispind, st:end, :])
    plt.title('Error Image')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
    plt.show()  # Only available for GUI
    plt.savefig(os.path.join(output_path, 'test_res.png'))
