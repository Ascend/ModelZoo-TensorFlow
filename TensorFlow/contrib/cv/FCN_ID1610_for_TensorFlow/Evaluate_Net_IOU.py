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

# Evaluate the perfomance of trained network by evaluating intersection over union (IOU) of the  network predcition
# and ground truth of the validation set
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input images for prediction are located
# 3) Set folder of ROI maps (for train images) in Label_Dir
#    the ROI Maps should be saved as png image with same name pixel of ROI should be 1 and all other pixels should be zero
# 4) Set folder for ground truth labels in Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# 5) Set number of classes number in NUM_CLASSES
# 6) Run script
###########################################################################################################################################################
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import IOU
import sys
import Data_Reader
import BuildNetVgg16
import CheckVGG16Model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", type=str, default="./")
parser.add_argument("--data_url", type=str, default="trainingfb/Materials_in_Vessels")
args = parser.parse_known_args()[0]
logs_dir=args.logs_dir
ROIMap_Dir=os.path.join(args.data_url,'VesselLabels')
Label_Dir=os.path.join(args.data_url,'LiquidSolidLabels')
Image_Dir=os.path.join(args.data_url,'Test_Images_All')
model_path=os.path.join(args.data_url,'vgg16.npy')
#......................................................................................................................................
#logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
#ROIMap_Dir="Data_Zoo/Materials_In_Vessels/VesselLabels/" # Folder where ROI map are save in png format (same name as coresponding image in images folder)
#Label_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
#Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Test image folder
#model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"

#-------------------------------------------------------------------------------------------------------------------------

Batch_Size=1
NUM_CLASSES = 4 #Number of classes
Classes = ["BackGround", "Empty Vessel","Liquid","Solid"] #List of classes
#VesseClasses=["Background","Vessel"] #Classes predicted for vessel region prediction
#PhaseClasses=["BackGround","Empty Vessel region","Filled Vessel region"]#

#ExactPhaseClasses=["BackGround","Vessel","Liquid","Liquid Phase two","Suspension", "Emulsion","Foam","Solid","Gel","Powder","Granular","Bulk","Bulk Liquid","Solid Phase two","Vapor"]
################################################################################################################################################################################
def main(argv=None):
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    # .........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    ROIMap = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="ROIMap")  # ROI input map
    # .........................Build FCN Net...............................................................................................
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class for the network
    Net.build(image, ROIMap, NUM_CLASSES, keep_prob)  # Create the net and load intial weights
    #    # -------------------------Data reader for validation image-----------------------------------------------------------------------------------------------------------------------------

    ValidReader = Data_Reader.Data_Reader(Image_Dir, ROIMap_Dir,GTLabelDir=Label_Dir, BatchSize=Batch_Size) # build reader that will be used to load images and labels from validation set

    #........................................................................................................................
    sess = tf.Session()  # Start Tensorflow session
    #--------Load trained model--------------------------------------------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    #if ckpt and ckpt.model_checkpoint_path:  # if train model exist restore it
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else: # if
        print("ERROR NO TRAINED MODEL IN: " + ckpt.model_checkpoint_path+"See TRAIN.py for training")
        sys.exit()
 #--------------------Sum of intersection from all validation images for all classes and sum of union for all images and all classes----------------------------------------------------------------------------------
    Union = np.float64(np.zeros(len(Classes))) #Sum of union
    Intersection =  np.float64(np.zeros(len(Classes))) #Sum of Intersection
    fim = 0
    print("Start Evaluating intersection over union for "+str(ValidReader.NumFiles)+" images")
 #===========================GO over all validation images and caclulate IOU============================================================
    while (ValidReader.itr<ValidReader.NumFiles):
        print(str(fim*100.0/ValidReader.NumFiles)+"%")
        fim+=1

#.........................................Run Predictin/inference on validation................................................................................
        Images, ROIMaps, GTLabels = ValidReader.ReadNextBatchClean()  # Read images ROI Map and ground truth annotation
        #Predict annotation using net
        PredictedLabels= sess.run(Net.Pred,feed_dict={image: Images, ROIMap:ROIMaps, keep_prob: 1.0})
#............................Calculate Intersection and union for prediction...............................................................

#        print("-------------------------IOU----------------------------------------")
        CIOU,CU=IOU.GetIOU(PredictedLabels,GTLabels.squeeze(),len(Classes),Classes) #Calculate intersection over union
        Intersection+=CIOU*CU
        Union+=CU

#-----------------------------------------Print results--------------------------------------------------------------------------------------
    print("---------------------------Mean Prediction----------------------------------------")
    print("---------------------IOU=Intersection Over Inion----------------------------------")
    for i in range(len(Classes)):
        if Union[i]>0: print(Classes[i]+"\t"+str(Intersection[i]/Union[i]))


##################################################################################################################################################
main()#Run script
print("Finished")