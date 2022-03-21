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
from utils import *
import glob
import scipy.io
from argparse import ArgumentParser
import pb_test
parser=ArgumentParser()
parser.add_argument('--modelpath', type=str, dest='modelpath', default='./Model/Model2/model-82500')
parser.add_argument('--inputpath', type=str, dest='inputpath', default='TestSet/Set5/g13/LR/')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='TestSet/Set5/GT_crop/')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='TestSet/Set5/g13/kernel.mat')
parser.add_argument('--savepath', type=str, dest='savepath', default='./output')
parser.add_argument('--ck2pb', dest='ck2pb', default=False, action='store_true')
parser.add_argument('--om2test', dest='om2test', default=False, action='store_true')
parser.add_argument('--png2bin', dest='png2bin', default=False, action='store_true')
args= parser.parse_args()

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.95

def main():

    if not os.path.exists('%s' % (args.savepath)):
        os.makedirs('%s' % (args.savepath))

    img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
    gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))
    scale=2.0

    try:
        kernel=scipy.io.loadmat(args.kernelpath)['kernel']
    except:
        kernel='cubic'


    Tester = pb_test.Test(args.modelpath, args.savepath, kernel, scale, conf, 0, 1)

    if (args.png2bin==True) or (args.om2test==True):
        for i in range(len(img_path)):
            img = imread(img_path[i])
            gt = imread(gt_path[i])

            Tester(img, gt, img_path[i])
        print('png2bin done')
    else:
        img = imread(img_path[0])
        gt = imread(gt_path[0])
        Tester(img, gt, img_path[0])


if __name__=='__main__':
    main()



# python freeze_graph.py  --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat --model 0
#  --num 1
'''
python freeze_graph.py  --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat  --ck2pb --png2bin'
'''