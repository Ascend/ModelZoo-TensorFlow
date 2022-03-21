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
# limitations under the License.import os
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
import time
import numpy as np
import tensorflow as tf

from dataset import Dataset
from inpaint_model import ModelInpaint
import poissonblending as poisson
import utils as utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags, self.flags.dataset)
        self.model = ModelInpaint(self.sess, self.flags)

        self._make_folders()
        self.iter_time = 0

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.test_out_dir = "{}/inpaint/{}/is_blend_{}".format(self.flags.dataset, self.flags.load_model,
                                                               str(self.flags.is_blend))
        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)

        self.train_writer = tf.summary.FileWriter("{}/inpaint/{}/is_blend_{}/{}/log".format(
            self.flags.dataset, self.flags.load_model, str(self.flags.is_blend), self.flags.mask_type),
            graph_def=self.sess.graph_def)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')
        
        for num_try in range(self.flags.num_try):
            self.model.preprocess()  # initialize memory for inpaint model

            imgs = self.dataset.val_next_batch(batch_size=self.flags.sample_batch)  # random select in validation data
            best_loss = np.ones(self.flags.sample_batch) * 1e10
            best_outs = np.zeros_like(imgs)

            start_time = time.time()  # measure inference time
            for iter_time in range(self.flags.iters):
                loss, img_outs, summary = self.model(imgs, iter_time)  # inference   #img_outs为dcgan.g_samples

                # save best gen_results accroding to the total loss
                for iter_loss in range(self.flags.sample_batch):
                    if best_loss[iter_loss] > loss[2][iter_loss]:  # total loss
                        best_loss[iter_loss] = loss[2][iter_loss]
                        best_outs[iter_loss] = img_outs[iter_loss]

                self.model.print_info(loss, iter_time, num_try)  # pring loss information

                if num_try == 0:  # save first try-information on the tensorboard only
                    self.train_writer.add_summary(summary, iter_time)  # write to tensorboard
                    self.train_writer.flush()

            blend_results = self.postprocess(imgs, best_outs, self.flags.is_blend)  # blending

            total_time = time.time() - start_time
            print('Total PT: {:.3f} sec.'.format(total_time))

            img_list = [(imgs + 1.) / 2., blend_results, (imgs + 1.) / 2.]
            self.model.plots(img_list, self.test_out_dir, num_try)  # save all of the images

    def postprocess(self, ori_imgs, gen_imgs, is_blend=True):     
        outputs = np.zeros_like(ori_imgs)
        tar_imgs = np.asarray([utils.inverse_transform(img) for img in ori_imgs])  # from (-1, 1) to (0, 1)
        sour_imgs = np.asarray([utils.inverse_transform(img) for img in gen_imgs])  # from (-1, 1) to (0, 1)

        if is_blend is True:
            for idx in range(tar_imgs.shape[0]):
                outputs[idx] = np.clip(poisson.blend(tar_imgs[idx], sour_imgs[idx],
                                                     ((1. - self.model.masks[idx]) * 255.).astype(np.uint8)), 0, 1)
        else:
            outputs = np.multiply(tar_imgs, self.model.masks) + np.multiply(sour_imgs, 1. - self.model.masks)

        return outputs

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')
            return True
        else:
            return False
