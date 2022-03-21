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
import ck2pb
import dataGenerator
import train
import test
from utils import *
from config import *
from tensorflow.core.protobuf.rewriter_config_pb2 import  RewriterConfig
import glob
import scipy.io

def main():
    if args.is_train==True:
        ##npu
        conf=tf.ConfigProto()
        custom_op = conf.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        #custom_op.parameter_map["modify_mixlist"].s=tf.compat.as_bytes("./ops_info.json")
        conf.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        conf.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

        data_generator=dataGenerator.dataGenerator(output_shape=[HEIGHT,WIDTH,CHANNEL], meta_batch_size=META_BATCH_SIZE,
                                                   task_batch_size=TASK_BATCH_SIZE,tfrecord_path=os.path.join(args.dataset, "train_SR_MZSR.tfrecord"))

        Trainer = train.Train(trial=args.trial, step=args.step, size=[HEIGHT,WIDTH,CHANNEL],
                              scale_list=SCALE_LIST, meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR, meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER, data_generator=data_generator, checkpoint_dir=CHECKPOINT_DIR, conf=conf)

        Trainer()
    else:
        ##cpu
        conf=tf.ConfigProto()

        img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
        gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))

        scale=2.0

        try:
            kernel=scipy.io.loadmat(args.kernelpath)['kernel']
        except:
            kernel='cubic'
        tmp_path = './SR/Model0/'
        tmp=-1
        index=0
        for j in range(2500,100000,2500):
            model_path=os.path.join(tmp_path,'model-'+str(j))
            Tester=test.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
            P=[]
            for i in range(len(img_path)):
                img=imread(img_path[i])
                gt=imread(gt_path[i])

                _, pp =Tester(img, gt, img_path[i])

                P.append(pp)

            avg_PSNR=np.mean(P, 0)

            print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))

            if avg_PSNR[1]>tmp:
                tmp = avg_PSNR[1]
                index=j
        print('better ： index :%d '%index,  'PSNR:  %.4f' %tmp)

        if args.is_ck2pb==True:
            pb=ck2pb.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
            img = imread(img_path[i])
            gt = imread(gt_path[i])
            pb(img, gt, img_path[i])



if __name__=='__main__':
    main()