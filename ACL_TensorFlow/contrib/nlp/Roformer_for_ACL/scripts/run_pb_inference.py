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

import tensorflow as tf
import numpy as np
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import os
import time
import argparse

np.random.seed(10)

def load_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")
    return graph

def NetworkRun(modelPath,inputPath,outputPath,batchsize,npu_predict):
    graph = load_graph(modelPath)
    input_node1 = graph.get_tensor_by_name('Input-Segment:0')
    input_node2 = graph.get_tensor_by_name('Input-Token:0')
    output_nodes = graph.get_tensor_by_name('Identity:0')
    cos_res = []
    with tf.Session(graph=graph) as sess:
        for file in os.listdir(os.path.join(inputPath,'Input-Segment'):
            if file.endswith(".bin"):
                input_1 = np.fromfile(os.path.jon(inputPath,"/Input-Segment",file),dtype="float32").reshape(batchsize,1024)
                input_2 = np.fromfile(os.path.jon(inputPath,"/Input-Token",file),dtype="float32").reshape(batchsize,1024)
                t0 = time.time()
                out = sess.run(output_nodes, feed_dict= {input_node1: input_1,input_node2: input_2})
                t1 = time.time()
                out.tofile(os.path.join(outputPath,"cpu_out_"+file))
                vec1 = np.fromfile(os.path.join(outputPath,"cpu_out_"+file),dtype="float32")
                vec2 = np.fromfile(npu_predict,"davinci_{}_output0.bin".format(file),dtype="float32")
                cos_sim = vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_res.append(cos_sim)
                print("Cosine Similarity of {}: {}".format(file,)cos_sim)
        print("Mean Cosine Similarity:{:.4f}".format(sum(cos_res)/len(cos_res)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model/wide_resnet.pb")
    parser.add_argument("--input", type=str, default="./input/")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--output", type=str, default="./cpu_output/")
    parser.add_argument("--npu_output", type=str, default="./npu_output/")
    args = parser.parse_args()
    if not os.path.exists(args.output):
            os.makedirs(args.output)
    NetworkRun(args.model,args.input,args.output,args.batchsize,args.npu_output)
