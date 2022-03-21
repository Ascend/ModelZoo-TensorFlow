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
import sys
import argparse
#import moxing as mox
from TrainTeacher import *
from TeacherPrediction import *
from trainStudent import  *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_url",type=str,default="./dataset")
    # parser.add_argument("--train_url",type=str,default="./output")
    parser.add_argument("--UseTeacherPrediction",type=bool,default=False)
    parser.add_argument("--studentlr",type=float,default=1e-04)
    parser.add_argument("--temperature",type=int,default=10)
    parser.add_argument("--fasttrain",type=bool,default=False)
    parser.add_argument("--trainlabel",type=str,default="./CK/label/train.txt")
    parser.add_argument("--testlabel",type=str,default="./CK/label/test.txt")
    parser.add_argument("--datapath",type=str,default="./CK/cohn-kanade")
    #add
    parser.add_argument("--pred_path",type=str,default="./outputPredictionPos")
    parser.add_argument("--InceptionV3_weight_path",type=str,default="./TeacherExpNet_CK.h5")
    parser.add_argument("--weights_path",type=str,default="./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
    parser.add_argument("--model_path",type=str,default="./result/OutModel/")
    parser.add_argument("--stepSize",type=int,default=50000)
    parser.add_argument("--epochs",type=int,default=3000)
    parser.add_argument("--testStep",type=int,default=20)
    parser.add_argument("--displayStep",type=int,default=20)
    parser.add_argument("--TrainTeacher",type=bool,default=False)
    parser.add_argument("--TeachervalPath",type=str,default="./CK/Teacher/test")
    parser.add_argument("--TeachertrainPath",type=str,default="./CK/Teacher/train")

    args=parser.parse_args()
    # data_dir="/cache/dataset"
    # model_dir="/cache/result"
    # graph_dir="/cache/result/Graph"
    # outmodel_dir="/cache/result/OutModel"
    # os.makedirs(data_dir)
    # os.makedirs(model_dir)
    # os.makedirs(graph_dir)
    # os.makedirs(outmodel_dir)
    #mox.file.copy_parallel(args.data_url,data_dir)
    if args.TrainTeacher:
        trainTeacher(args.TeachervalPath,args.TeachertrainPath,"result/Graph/teacher.png")
    if args.UseTeacherPrediction:
        teacherPrediction(args.datapath,args.pred_path,args.InceptionV3_weight_path,args.weights_path)
    trainStudent(args.testlabel,args.trainlabel,"result/Graph/student.png",args.model_path+"ckmodel.ckpt",
                args.studentlr,args.temperature,args.fasttrain,args.stepSize,args.epochs,args.testStep,args.displayStep,args.datapath )
    #mox.file.copy_parallel(model_dir, args.train_url)