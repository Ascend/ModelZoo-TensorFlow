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

from npu_bridge.npu_init import *
import tensorflow as tf

from SRFBN_model import SRFBN
from PreProcess import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import _structural_similarity




def test_SRFBN(image_lr,image_hr):
    
    #image
    height, width, _ = image_lr.shape
    print(height,width)
    global load_flag
    global srfbn
    global out
    if load_flag == 0:
        srfbn = SRFBN(sess, cfg)
        out = srfbn.test(width, height)
        tf.global_variables_initializer().run(session=sess)
        srfbn.saver = tf.train.Saver(max_to_keep=1)
        srfbn.load()
        srfbn.l2_regularization_loss = tf.reduce_sum(tf.get_collection("weights_l2_loss"))
        srfbn.losses = [srfbn.calc_loss(x=x, y=srfbn.labelplaceholder, loss_type=srfbn.cfg.loss_type) for x in
                        srfbn.outs]
        srfbn.losses = tf.reduce_sum(srfbn.losses) / len(srfbn.losses) / srfbn.cfg.batchsize + srfbn.l2_regularization_loss
        load_flag += 1
    #cv2.namedWindow("result", 0)
    
    img_hr = image_hr.reshape([1,height*srfbn.cfg.scale,width*srfbn.cfg.scale,3])
    img_lr = image_lr.reshape([1, height, width, 3])
    output,err,l2_loss = sess.run([out,srfbn.losses,srfbn.l2_regularization_loss], feed_dict={srfbn.imageplaceholder: img_lr,srfbn.labelplaceholder:img_hr})
    output = output[0] * 128 + 127.5
    img_hr = img_hr.reshape([height*srfbn.cfg.scale,width*srfbn.cfg.scale,3])
    PSNR = compare_psnr(output, img_hr, data_range=255)
    ssim = _structural_similarity.structural_similarity(output, img_hr,win_size=11, data_range=255, multichannel=True)
    print("loss:[%.8f], l2_loss:[%.8f], PSNR:[%.8f], SSIM:[%.8f]"%(err,l2_loss,PSNR,ssim))
    #cv2.imshow("result", np.uint8(output))
    #cv2.waitKey(0)
    


if __name__ == '__main__':
    sess = tf.Session(config=npu_config_proto())
    from config import SRFBN_config
    cfg = SRFBN_config()
    cfg.istest = True
    cfg.istrain = False
    image = "/home/TestUser08/BUAA/Resolution_2K/DIV2K/DIV2K_valid_HR/0801.png"
    batch_label,batch_lrimage = preprocess([image,],cfg)
    batch_lrimage = np.array(batch_lrimage)
    batch_label = np.array(batch_label)
    load_flag = 0
    for i in range(batch_label.shape[0]):
        test_SRFBN(batch_lrimage[i],batch_label[i])
    srfbn.sess.close()

