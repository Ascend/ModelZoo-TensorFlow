
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

#coding=utf-8
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import Data_Reader
import BuildNetVgg16
import os
import CheckVGG16Model
import time
import scipy.misc as misc

# 从obs导入数据
import argparse
#import moxing as mox
# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="trainingfb/Materials_in_Vessels")
parser.add_argument("--logs_dir", type=str, default="./log")
parser.add_argument("--UseValidationSet", type=str, default=False)
parser.add_argument("--MAX_ITERATION", type=int, default=100000)
args = parser.parse_known_args()[0]
# 在ModelArts容器创建数据存放目录
#data_dir = "/cache/dataset"
#os.makedirs(data_dir)
# OBS数据拷贝到ModelArts容器内
#mox.file.copy_parallel(config.data_url, data_dir)

# 创建session添加代码
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

#...........................................Input and output folders.................................................
#Train_Image_Dir="/home/ma-user/modelarts/inputs/data_url_0/Train_Images/" # Images and labels for training
#ROIMap_Dir="/home/ma-user/modelarts/inputs/data_url_0/VesselLabels/" # Folder where ROI map are save in png format (same name as coresponding image in images folder)
#Label_Dir="/home/ma-user/modelarts/inputs/data_url_0/LiquidSolidLabels/"# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
UseValidationSet=args.UseValidationSet# do you want to use validation set in training
#Valid_Image_Dir="/home/ma-user/modelarts/inputs/data_url_0/Test_Images_All/"# Validation images that will be used to evaluate training (the ROImap and Labels are in same folder as the training set)
#logs_dir= "/home/ma-user/modelarts/user-job-dir/code/logs/"# "path to logs directory where trained model and information will be stored"
Train_Image_Dir=os.path.join(args.data_url,'Train_Images')
ROIMap_Dir=os.path.join(args.data_url,'VesselLabels')
Label_Dir=os.path.join(args.data_url,'LiquidSolidLabels')
Valid_Image_Dir=os.path.join(args.data_url,'Test_Images_All')
logs_dir=args.logs_dir

if not os.path.exists(logs_dir): os.makedirs(logs_dir)
#model_path="/home/ma-user/modelarts/user-job-dir/code/Model_Zoo/vgg16.npy"
model_path=os.path.join(args.data_url,'vgg16.npy')
learning_rate=1e-5 #Learning rate for Adam Optimizer
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
#-----------------------------Other Paramters------------------------------------------------------------------------
TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Batch_Size=2# Number of files per training iteration
Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = args.MAX_ITERATION # Max  number of training iteration
NUM_CLASSES = 4#Number of class for fine grain +number of class for solid liquid+Number of class for empty none empty +Number of class for vessel background
######################################Solver for model   training#####################################################################################################################

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
################################################################################################################################################################################
def main(argv=None):
    tf.reset_default_graph()
    keep_prob= tf.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    ROIMap= tf.placeholder(tf.int32, shape=[None, None, None, 1], name="ROIMap")  # ROI input map
    GTLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")#Ground truth labels for training
  #.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    Net.build(image, ROIMap,NUM_CLASSES,keep_prob)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................

    Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(GTLabel, squeeze_dims=[3]), logits=Net.Prob,name="Loss")))  # Define loss function for training

   #....................................Create solver for the net............................................................................................
    trainable_var = tf.trainable_variables() # Collect all trainable variables for the net
    train_op = train(Loss, trainable_var) #Create Train Operation for the net
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir, ROIMap_Dir, GTLabelDir=Label_Dir,BatchSize=Batch_Size) #Reader for training data
    if UseValidationSet:
        ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir, ROIMap_Dir, GTLabelDir=Label_Dir,BatchSize=Batch_Size) # Reader for validation data

    # sess = tf.Session() #Start Tensorflow session
    # -------------load trained model if exist-----------------------------------------------------------------
    sess = tf.Session(config=config)
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer()) #Initialize variables
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------

    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tloss\t Learning Rate="+str(learning_rate))
    f.close()
    if UseValidationSet:
       f = open(ValidLossTxtFile, "w")
       f.write("Iteration\tloss\t Learning Rate=" + str(learning_rate))
       f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        Images, ROIMaps, GTLabels =TrainReader.ReadNextBatchClean() # Load  augmeted images and ground true labels for training
        feed_dict = {image: Images,GTLabel:GTLabels,ROIMap:ROIMaps, keep_prob: 0.5}
        sess.run(train_op, feed_dict=feed_dict) # Train one cycle
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 1000 == 0 and itr>0:
            print("Saving Model to file in"+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) #Save model
#......................Write and display train loss..........................................................................
        if itr % 100==0:
            # Calculate train loss
            feed_dict = {image: Images, GTLabel: GTLabels, ROIMap: ROIMaps, keep_prob:0.5}
            t1=time.time()
            # from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
            # config = tf.ConfigProto()
            # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
            # custom_op.name = "NpuOptimizer"
            # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
            # sess = tf.Session(config=config)
            TLoss=sess.run(Loss, feed_dict=feed_dict)
            t2=time.time()
            print("Step "+str(itr)+" Train Loss="+str(TLoss))
            print("时间\n", t2-t1,'s/100itr')
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(TLoss))
                f.close()
#......................Write and display Validation Set Loss by running loss on all validation images.....................................................................
        if UseValidationSet and itr % 2000 == 0:
            SumLoss=np.float64(0.0)
            NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
            print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
            for i in range(NBatches):# Go over all validation image
                Images, ROIMaps,GTLabels= ValidReader.ReadNextBatchClean() # load validation image and ground true labels
                feed_dict = {image: Images,ROIMap:ROIMaps, GTLabel: GTLabels ,keep_prob: 1.0}
                # Calculate loss for all labels set
                TLoss = sess.run(Loss, feed_dict=feed_dict)
                SumLoss+=TLoss
                NBatches+=1
            SumLoss/=NBatches
            print("Validation Loss: "+str(SumLoss))
            with open(ValidLossTxtFile, "a") as f:
                f.write("\n" + str(itr) + "\t" + str(SumLoss))
                f.close()
##################################################################################################################################################

main()
print("Finished")
