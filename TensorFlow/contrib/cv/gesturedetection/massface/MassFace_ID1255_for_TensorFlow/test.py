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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.insert(0,'networks')
sys.path.insert(0,'lib')
from lib import utils
from lib import lfw
import os
import moxing as mox
import math
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
#from models import resnet_v1, resnet_v2,resnet_v1_modify,resnet_v2_modify
from networks import sphere_network as network
from networks import MobileFaceNet as mobilenet
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import importlib
import pdb

def main(args):
    args.lfw_dir = "/home/ma-user/modelarts/user-job-dir/MassFac/dataset/lfw-112x112/"
    args.lfw_pairs = "/home/ma-user/modelarts/user-job-dir/MassFac/dataset/pairs.txt"
    args.model = "/home/ma-user/modelarts/user-job-dir/MassFac/models/20210923-222923/model-20210923-222923.ckpt-600"
    with tf.Graph().as_default():
    #!with tf.device("/cpu:0"):
        #!config = tf.ConfigProto()
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        #config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:

            # Read the file containing the pairs used for testing
            #pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
            #pdb.set_trace()

            # Get the paths for the corresponding images
            #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            #paths,actual_issame = lfw.get_paths_and_pairs(os.path.expanduser(args.lfw_dir),args.lfw_pairs)
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
            #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), args.lfw_pairs)
            #pdb.set_trace()



            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            print('image size',image_size)
            #images_placeholder = tf.placeholder(tf.float32,shape=(None,image_size,image_size,3),name='image')
            images_placeholder = tf.placeholder(tf.float32,shape=(None,args.image_height,args.image_width,3),name='image')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            #with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
            if args.network_type == 'resnet50':
                with slim.arg_scope(resnet_v2.resnet_arg_scope(0.0005)):
                    prelogits, end_points = resnet_v2.resnet_v2_50(images_placeholder,is_training=phase_train_placeholder,num_classes=args.embedding_size,output_stride=16)
                    #prelogits, end_points = resnet_v2.resnet_v2_50(images_placeholder,is_training=phase_train_placeholder,num_classes=256,output_stride=8)
                    #prelogits, end_points = resnet_v2_modify.resnet_v2_50(images_placeholder,is_training=phase_train_placeholder,num_classes=256)
                    #prelogits = slim.batch_norm(prelogits, is_training=phase_train_placeholder,epsilon=1e-5, scale=True,scope='softmax_bn')
                    prelogits = tf.squeeze(prelogits,[1,2],name='SpatialSqueeze')

            elif args.network_type == 'sphere_network':
                prelogits = network.infer(images_placeholder,args.embedding_size)
                if args.fc_bn:
                    print('do batch norm after network')
                    prelogits = slim.batch_norm(prelogits, is_training=phase_train_placeholder,epsilon=1e-5, scale=True,scope='softmax_bn')

            elif args.network_type ==  'mobilenet':
                prelogits, net_points = mobilenet.inference(images_placeholder,bottleneck_layer_size=args.embedding_size,phase_train=phase_train_placeholder,weight_decay=0.0005,reuse=False)



            #embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            embeddings = tf.identity(prelogits)
            #saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            saver.restore(sess, args.model)
            if args.save_model:
                saver.save(sess,'./tmp_saved_model',global_step=1)
                return 0

            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            if args.do_flip:
                embedding_size *= 2
                emb_array = np.zeros((nrof_images, embedding_size))
            else:
                emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                print('handing {}/{}'.format(start_index,nrof_images))
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                #images = facenet.load_data(paths_batch, False, False, image_size,True,image_size)
                #images = facenet.load_data2(paths_batch, False, False, args.image_height,args.image_width,True,)
                images = utils.load_data(paths_batch, False, False, args.image_height,args.image_width,args.prewhiten,(args.image_height,args.image_width))
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feats = sess.run(embeddings, feed_dict=feed_dict)
                if args.do_flip:
                    images_flip = utils.load_data(paths_batch, False, True, args.image_height,args.image_width,args.prewhiten,(args.image_height,args.image_width))
                    feed_dict = { images_placeholder:images_flip, phase_train_placeholder:False }
                    feats_flip = sess.run(embeddings, feed_dict=feed_dict)
                    feats = np.concatenate((feats,feats_flip),axis=1)
                    #feats = (feats+feats_flip)/2
                #images = facenet.load_data(paths_batch, False, False, 160,True,182)
                #images = facenet.load_data(paths_batch, False, False, image_size,src_size=256)
                #feed_dict = { images_placeholder:images, phase_train_placeholder:True}
                #pdb.set_trace()
                #feats = facenet.prewhiten(feats)
                feats = utils.l2_normalize(feats)
                emb_array[start_index:end_index,:] = feats
                #pdb.set_trace()

            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='/home/ma-user/modelarts/user-job-dir/MassFac/dataset/lfw-112x112')
    parser.add_argument('--network_type', type=str,
        help='Network structure.',default='mobilenet ')
    parser.add_argument('--fc_bn', 
        help='wheather bn is followed by fc layer.',default=False,action='store_true')
    parser.add_argument('--prewhiten', 
        help='wheather do prewhiten to preprocess image.',default=False,action='store_true')
    parser.add_argument('--save_model', type=bool,
        help='whether save model to disk.',default=False)
    parser.add_argument('--do_flip', type=bool,
        help='wheather flip is used in test.',default=False)
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=200)
    parser.add_argument('--embedding_size', type=int,
        help='Feature embedding size.', default=1024)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='/home/ma-user/modelarts/user-job-dir/MassFac/models/20210923-222923/model-20210923-222923.ckpt-600')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='./dataset/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    #parser.add_argument('--model_def', type=str,
        #help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    #parser.add_argument("--modelarts_data_dir", type=str, default="/home/ma-user/modelarts/user-job-dir/MassFac/dataset/")
    #parser.add_argument("--modelarts_result_dir", type=str, default="/home/ma-user/modelarts/user-job-dir/MassFac/result")

    return parser.parse_args(argv)

if __name__ == '__main__':

    #mox.file.copy_parallel(src_url="obs://qyy-unet/MassFace-master/dataset/lfw-112x112/", dst_url="data")
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{},work_dir:{}".format(code_dir,work_dir))
    print(os.listdir(os.getcwd()))

    config = parse_arguments(sys.argv[1:])
    print("--------config--------")
    for k in list(vars(config).keys()):
        print("keys:{}:value:{}".format(k, vars(config)[k]))
    print("--------config--------")

    #if not os.path.exists(config.modelarts_result_dir):
        #os.makedirs(config.modelarts_result_dir)
        #bash_header = os.path.join(code_dir, 'train_triplet.sh')
        #arg__url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
        #bash_command = 'bash %s %s' % (bash_header, arg__url)
        #print("bash command:", bash_command)
        #os.system(bash_command)

    main(parse_arguments(sys.argv[1:]))
