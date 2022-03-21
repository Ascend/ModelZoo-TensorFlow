
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

import model
import time
import imageio
from utils import *
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from pathlib import Path
import numpy
from freeze_graph import args
class Test(object):
    def __init__(self, model_path, save_path,kernel, scale, conf, method_num, num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        self.max_iters=num_of_adaptation
        self.display_iter = 1

        self.upscale_method= 'cubic'
        self.noise_level = 0.0

        self.back_projection=False
        self.back_projection_iters=4

        self.model_path=model_path
        self.save_path=save_path
        self.method_num=method_num

        self.ds_method=methods[self.method_num]

        self.kernel = kernel
        self.scale=scale
        self.scale_factors = [self.scale, self.scale]
        self.sess = tf.Session(config=conf)
        self.build_network(conf)

    def build_network(self, conf):
        tf.reset_default_graph()

        self.lr_decay = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        # Input image
        self.input= tf.placeholder(tf.float32, shape=[None,None,None,3], name='input')
        # Ground truth
        self.label = tf.placeholder(tf.float32, shape=[None,None,None,3],  name='label')

        # parameter variables
        self.PARAM=model.Weights(scope='MODEL')
        # model class (without feedforward graph)
        self.MODEL = model.MODEL(name='MODEL')
        # Graph build
        self.MODEL.forward(self.input,self.PARAM.weights)
        self.output=self.MODEL.output

        self.loss_t = tf.losses.absolute_difference(self.label, self.output)

        # Optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr_decay).minimize(self.loss_t)
        self.init = tf.global_variables_initializer()

        # Variable lists
        self.var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MODEL')

        self.loader=tf.train.Saver(var_list=self.var_list)

        self.sess=tf.Session(config=conf)

    def initialize(self):
        self.sess.run(self.init)
        ##加载ckpt
        self.loader.restore(self.sess, self.model_path)
        # ####加载pb，和运行pb配套使用，注意注释的代码，需要main函数输入图片循环变为1，不然模型冻结后，在训练可能没有效果，故而冻结代码在训练后执行
        # with gfile.FastGFile('./output_model/pb_model_npu/frozen_model.pb', 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     self.sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')
        # print('=============== Load Meta-trained Model parameters... ==============')

        self.loss = [None] * self.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.psnr=[]
        self.iter = 0

    def __call__(self, img, gt, img_name):
        self.img=img
        self.gt = modcrop(gt, self.scale)

        self.img_name=img_name

        print('** Start Adaptation for X', self.scale, os.path.basename(self.img_name), ' **')
        # Initialize network
        self.initialize()

        self.sf = np.array(self.scale_factors)
        self.output_shape = np.uint(np.ceil(np.array(self.img.shape[0:2]) * self.scale))

        # # Train the network
        self.quick_test()

        if args.png2bin==False:
            print('[*] Baseline ')
            self.train()
            if args.ck2pb==True:
                save(tf.train.Saver(max_to_keep=10000), self.sess, args.savepath, 0, 0)
                # print(self.sess.run('MODEL/conv8/bias/read:0'))
                # #冻结pb
                tf.train.write_graph(self.sess.graph_def, args.savepath, 'model.pb')
                # 将模型参数与模型图结合，并保存为pb文件
                # freeze_graph.freeze_graph('output_model/pb_model_npu_1/model.pb', '', False, './output_model/pb_model_npu_1/Model0/model-0', 'MODEL_1/output',
                #                           'save/restore_all', 'save/Const:0', 'output_model/pb_model_npu_1/frozen_model.pb', False,
                #                           "")
                freeze_graph.freeze_graph(os.path.join(args.savepath,'model.pb'), '', False,
                                          os.path.join(args.savepath, 'Model0/model-0') , 'MODEL_1/output',
                                          'save/restore_all', 'save/Const:0', os.path.join(args.savepath,'frozen_model.pb'),
                                          False,
                                          "")
                # print(self.sess.run('MODEL/conv8/bias/read:0'))
                print('ck2pb done')
            if args.om2test==True:
                self.final_test()


    def train(self):
        self.hr_father = self.img
        self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
        self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)

        t1=time.time()
        for self.iter in range(self.max_iters):

            if self.method_num == 0:
                '''direct'''
                if self.iter==0:
                    self.learning_rate=2e-2
                elif self.iter < 4:
                    self.learning_rate=1e-2
                else:
                    self.learning_rate=5e-3

            elif self.method_num == 1:
                '''Multi-scale'''
                if self.iter < 3:
                    self.learning_rate=1e-2
                else:
                    self.learning_rate=5e-3

            elif self.method_num == 2:
                '''bicubic'''
                if self.iter == 0:
                    self.learning_rate = 0.01
                elif self.iter < 3:
                    self.learning_rate = 0.01
                else:
                    self.learning_rate = 0.001

            elif self.method_num == 3:
                ''''scale 4'''
                if self.iter ==0:
                    self.learning_rate=1e-2
                elif self.iter < 5:
                    self.learning_rate=5e-3
                else:
                    self.learning_rate=1e-3

            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father)

            # Display information
            if self.iter % self.display_iter == 0:
                print('Scale: ', self.scale, ', iteration: ', (self.iter+1), ', loss: ', self.loss[self.iter])

            # Test network during adaptation

            # if self.iter % self.display_iter == 0:
            #     output=self.quick_test()

            # if self.iter==0:
            #     imageio.imsave('%s/%02d/01/%s.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4]), output)
            # if self.iter==9:
            #     imageio.imsave('%s/%02d/10/%s_%d.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4], self.iter), output)

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))

    def forward_pass(self, input, output_shape=None):
        ILR = imresize(input, self.scale, output_shape, self.upscale_method)
        feed_dict = {self.input : ILR[None,:,:,:]}
        output_=self.sess.run(self.output, feed_dict)

        # # # 运行pb模型
        # node_in = self.sess.graph.get_tensor_by_name('input:0')  # 此处填入输入节点名称
        # node_out = self.sess.graph.get_tensor_by_name('MODEL_1/output:0')  # 此处填入输出节点名称
        # feed_dict = {node_in: ILR[None, :, :, :]}
        # output_ = self.sess.run(node_out, feed_dict)
        # print(Path(self.img_name).stem)


        # #输入图片转bin，为om输入做准备
        if args.png2bin==True:
            name = Path(self.img_name).stem
            tmp = ILR[None, :, :, :]
            tmp = np.asarray(tmp, dtype='float32')

            # print(ILR.shape)
            tmp.tofile(name+'.bin')

        return np.clip(np.squeeze(output_), 0., 1.)

    def forward_backward_pass(self, input, hr_father):
        ILR = imresize(input, self.scale, hr_father.shape, self.upscale_method)

        HR = hr_father[None, :, :, :]

        # Create feed dict
        feed_dict = {self.input: ILR[None,:,:,:], self.label: HR, self.lr_decay: self.learning_rate}

        # Run network
        _, self.loss[self.iter], train_output = self.sess.run([self.opt, self.loss_t, self.output], feed_dict=feed_dict)
        return np.clip(np.squeeze(train_output), 0., 1.)

    def hr2lr(self, hr):
        lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
        return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)

    def quick_test(self):
        # 1. True MSE
        self.sr = self.forward_pass(self.img, self.gt.shape)


        self.mse = self.mse + [np.mean((self.gt - self.sr)**2)]

        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])

        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)), rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8)))
        self.psnr.append(PSNR)

        # 2. Reconstruction MSE
        if args.png2bin==False:
            self.reconstruct_output = self.forward_pass(self.hr2lr(self.img), self.img.shape)
            self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))

            processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)

            print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)

            return processed_output

    def final_test(self):

        # output = self.forward_pass(self.img, self.gt.shape)
        # if self.back_projection == True:
        #     for bp_iter in range(self.back_projection_iters):
        #         output = back_projection(output, self.img, down_kernel=self.kernel,
        #                                           up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)
        #
        # processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)
        #
        # '''Shave'''
        scale=int(self.scale)
        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
        #           rgb2y(processed_output)[scale:-scale, scale:-scale])



        #测试om输出的txt的精度，使用时记得找到此输出位置，仅仅对bird.png图片测试，其他图片大小不符合om的input shape

        print( Path(self.img_name).stem)
        tmp = numpy.loadtxt(args.savepath+'/'+Path(self.img_name).stem+'.txt')
        if Path(self.img_name).stem=='baby':
            tmp = tmp[0:512 * 512, :]
            out = numpy.reshape(tmp, (512, 512, 3)) #baby.png

        if Path(self.img_name).stem=='bird':
            tmp = tmp[0:288 * 288, :]
            out = numpy.reshape(tmp, (288, 288, 3))  # bird.png

        if Path(self.img_name).stem == 'butterfly':
            tmp = tmp[0:256 * 256, :]
            out = numpy.reshape(tmp, (256, 256, 3))  # butterfly.png

        if Path(self.img_name).stem == 'head':
            tmp = tmp[0:280 * 280, :]
            out = numpy.reshape(tmp, (280, 280, 3))  # head.png

        if Path(self.img_name).stem == 'woman':
            tmp = tmp[0:228 * 344, :]
            out = numpy.reshape(tmp, (344, 228, 3))  # woman.png
        # print((self.gt.shape))
        proce_out = np.round(np.clip(out * 255, 0., 255.)).astype(np.uint8)
        PSNR = psnr(rgb2y(np.round(np.clip(self.gt * 255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                    rgb2y(proce_out)[scale:-scale, scale:-scale])
        print(PSNR)
        return  int(PSNR)





        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
        #           rgb2y(processed_output))

        # self.psnr.append(PSNR)
        #
        # return processed_output






    ##pb转om ，在华为给的镜像上输入即可

    ## atc --input_format=NHWC --input_shape="1,288,288,3" --check_report=/root/MZSR/network_analysis.report --input_format=NHWC --output="/root/MZSR/mzsr_model" --soc_version=Ascend310 --framework=3 --model="/root/MZSR/frozen_model.pb" --out_nodes="MODEL_1/output:0"

    ##运行om模型

    ##./msame --model "/root/MZSR/mzsr_model.om"  --output "/root/MZSR/" --outfmt TXT --loop 1 --input "/root/MZSR/bird.bin"
# atc --model="/root/MZSR/frozen_model.pb" --framework=3 --output="/root/MZSR/mzsr_model" --soc_version=Ascend310 --input_shape="input:1,3,288,288" --log=info --out_nodes="MODEL_1/output:0"



# atc --model="/root/MZSR/frozen_model.pb" --framework=3 --output="/root/MZSR/mzsr_model" --soc_version=Ascend310 --input_shape="input:1,-1,-1,3"  --dynamic_dims="512,512; 288,288;256,256;280,280;228,344" --log=info --out_nodes="MODEL_1/output:0"