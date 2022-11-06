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
from SRFBN_model import SRFBN
import tensorflow as tf
import time
from PreProcess import *
from skimage.metrics import peak_signal_noise_ratio as comparepsnr
from skimage.metrics import _structural_similarity

def train_SRFBN(dataset, sess, cfg):
    # start put data in queue
    with tf.device('/cpu:0'):
        step = tf.Variable(0, trainable=False)
    srfbn = SRFBN(sess=sess, cfg=cfg)
    srfbn.train_step()
    out = tf.add_n(srfbn.outs) / srfbn.cfg.num_steps
    ## build Optimizer
    #使学习率在不同迭代阶段不同
    boundaries = [len(dataset)*epoch//cfg.batchsize for epoch in cfg.lr_steps]
    values = [cfg.learning_rate*(cfg.lr_gama**i) for i in range(len(cfg.lr_steps)+1)]
    lr = tf.train.piecewise_constant(step, boundaries, values)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies(update_ops):
        gs_vs = optimizer.compute_gradients(srfbn.losses)
        with tf.device('/cpu:0'):
            train_op = optimizer.apply_gradients(grads_and_vars=gs_vs, global_step=step)

    tf.global_variables_initializer().run(session=sess)

    summary_writer = tf.summary.FileWriter(cfg.srfbn_logdir, srfbn.sess.graph)
    #加载模型
    if srfbn.cfg.load_premodel:
        counter = srfbn.load()
    else:
        counter = 0
    time_ = time.time()
    print("\nNow Start Training...\n")
    global_step = 0
    for ep in range(cfg.epoch):
        
        #每次训练时挑选照片的顺序是随机的
        pic_idx = np.random.permutation(len(dataset))
        picid = 0
        #一次加载五张图像
        for i in range(0,len(dataset),5):
            index = []
            for j in range(5):
               index.append(pic_idx[i+j])
            imgnames =  []
            for pic in index:
                imgnames.append(dataset[pic])
            picid += 5 
            print(imgnames)
            batch_labels, batch_images = preprocess(imgnames, cfg)
            patch_idx = list(range(len(batch_labels)))
            #使得图片块的数量刚好能被batchsize整除
            if len(patch_idx) % cfg.batchsize != 0:
                patch_idx.extend(list(np.random.choice(patch_idx,
                                                   cfg.batchsize * ((len(patch_idx) // cfg.batchsize)+1) - len(patch_idx))))
            

            patch_idx = np.random.permutation(patch_idx)
            

            iterations = len(patch_idx) // cfg.batchsize
            

            for it in range(iterations):
                
                idx = list(patch_idx[it * cfg.batchsize: (it+1)* cfg.batchsize])
                

                patch_labels =  np.array(batch_labels)[idx]

                patch_images =  np.array(batch_images)[idx]


                output,_, loss,l2_loss,= srfbn.sess.run([out,train_op, srfbn.losses,srfbn.l2_regularization_loss],
                                      feed_dict={srfbn.imageplaceholder: patch_images,
                                                 srfbn.labelplaceholder: patch_labels})
                output = output[0] * 128 + 127.5
                img_hr = patch_labels.reshape([srfbn.cfg.imagesize * srfbn.cfg.scale, srfbn.cfg.imagesize * srfbn.cfg.scale, 3])
                PSNR = comparepsnr(output, img_hr, data_range=255)
                ssim = _structural_similarity.structural_similarity(output, img_hr, win_size=11, data_range=255,
                                                                    multichannel=True)
                
                if it % 10 == 0:
                    print("Epoch:%2d, pic:%d, step:%2d, global_step:%d, time :%4.4f, loss:%.8f, l2_loss:%.8f, PSNR:%.8f, SSIM:%.8f" % (
                        (ep + 1),picid, it,global_step,time.time() - time_, loss,l2_loss,PSNR,ssim))
                if it % 100 == 0:
                    srfbn.save(counter)
                    summary_str = srfbn.sess.run(srfbn.merged_summary,
                                                 feed_dict={srfbn.imageplaceholder: patch_images,
                                                            srfbn.labelplaceholder: patch_labels})
                    summary_writer.add_summary(summary_str, counter)

                global_step += 1
                counter += 1

#训练
def train(*args, **kwargs):
    data_dir = kwargs["data_dir"]
    imgs = [os.path.join(data_dir,data) for data in os.listdir(data_dir)]



    sess = tf.compat.v1.Session(config=npu_config_proto())

    ## build NetWork
    from config import SRFBN_config
    cfg = SRFBN_config()
    datasetet = imgs
    train_SRFBN(datasetet, sess, cfg)



if __name__ == '__main__':
    import os
    data_dir = "/home/TestUser08/BUAA/output_npu_20221021153629/SRFBN-tensorflow_npu_20221021153629/Resolution_2K/DIV2K/DIV2K_train_HR"
    train(data_dir=data_dir)

