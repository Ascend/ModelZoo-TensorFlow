'''
@inproceedings{cugu2019microexpnet,
  title={MicroExpNet: An Extremely Small and Fast Model For Expression Recognition From Face Images},
  author={Cugu, Ilke and Sener, Eren and Akbas, Emre},
  booktitle={2019 Ninth International Conference on Image Processing Theory, Tools and Applications (IPTA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
'''
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


from __future__ import print_function

import json
from time import gmtime, strftime
from Candidates.Preprocessing import *
from  MicroExpNet import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np
import cv2
import sys
import os
from npu_bridge.npu_init import *
import time

def trainStudent(testset,trainset,outputgraphname,outputmodelname,lr,temperature,fasttrain,stepSize,epochs,testStep,displayStep,datapath):
    # Static parameters
    imgXdim = 84
    imgYdim = 84
    nInput = imgXdim * imgYdim  # Since RGB is transformed to Grayscale
    nClasses = 8
    dropout = 0.95
    batchSize = 64
    # learningRate = 1e-04
    # stepSize = 50000
    # epochs = 3000
    # testStep = 20
    # displayStep = 20
    use10fold = 0
    '''
        testSet					: Index of the chosen test batch (10 batches in total) or file path of the test labels
        labelPath				: Absolute path of the label file
        outputGraphName			: Name of the learning curve graph
        outputModelName			: Name of the Tensorflow model file
        temperature		: Model compression parameter
     '''
    # Dynamic parameters
    testSet = testset
    trainSet = trainset
    outputGraphName = outputgraphname
    outputModelName = outputmodelname
    learningRate = lr
    temperature = temperature
    teacherPredPath = "./outputPredictionPos/preds.json" #add
    if fasttrain:
        epochs = 1000

    # Deploy images and their labels
    print("[" + get_time() + "] " + "Deploying images...")
    fp = open(teacherPredPath, 'r')
    data = fp.read()
    js = json.loads(data)

    fp.close()
    # Produce one-hot labels
    print("[" + get_time() + "] " + "Producing one-hot labels...")

    print("[" + get_time() + "] " + "Start training for val[" + str(testSet) + "]")

    print("[" + get_time() + "] " + "Initializing batches...")
    batches = []
    test_batches = []
    trainX, trainY, teacherLogits = deployImages(trainSet, js, datapath)  #add
    testX, testY, testT = deployImages(testSet, js, datapath)  #add
    trainY = produceOneHot(trainY, nClasses)
    testY = produceOneHot(testY, nClasses)
    batches.extend(produceBatch(trainX, trainY, teacherLogits, batchSize))
    test_batches.extend(produceBatch(testX, testY, testT, batchSize))

    print("[" + get_time() + "] " + "Initializing placeholders...")

    # tf Graph input
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, nInput],name='input')
    lr = tf.compat.v1.placeholder(tf.float32)
    keepProb = tf.compat.v1.placeholder(tf.float32)
    y = tf.compat.v1.placeholder(tf.int32, shape=[None, nClasses])
    teacher = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, nClasses])
    lambda_ = tf.compat.v1.placeholder(tf.float32)

    # Loss values for plotting
    train_loss_vals = []
    train_acc_vals = []
    train_iter_num = []
    test_loss_vals = []
    test_acc_vals = []
    test_iter_num = []
    fin_accuracy = 0
    classifier = None
    fp = open("./result/perf.txt", 'w')  #add

    # Construct model
    classifier = MicroExpNet(x, y, teacher, lr, nClasses, imgXdim, imgYdim, batchSize, dropout, temperature, lambda_)

    # Deploy weights and biases for the model saver
    model_saver = tf.compat.v1.train.Saver()
    weights_biases_deployer = tf.compat.v1.train.Saver({"wc1": classifier.w["wc1"], \
                                                        "wc2": classifier.w["wc2"], \
                                                        "wfc": classifier.w["wfc"], \
                                                        "wo": classifier.w["out"], \
                                                        "bc1": classifier.b["bc1"], \
                                                        "bc2": classifier.b["bc2"], \
                                                        "bfc": classifier.b["bfc"], \
                                                        "bo": classifier.b["out"]})
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    perf_list = []
    fps_list = []
    with tf.compat.v1.Session(config=config) as sess:
        # Initializing the variables
        sess.run(tf.compat.v1.global_variables_initializer())
        print("[" + get_time() + "] " + "Training is started...")
        step = 0
        # Keep training until each max iterations
        while step <= epochs:   #3000
            total_batch = len(batches)
            total_test_batch = len(test_batches)
            for i in range(total_batch):  #108
                start = time.time()
                batch_x = batches[i]['x']
                batch_y = batches[i]['y']
                batch_teacher = batches[i]['teacherLogits']
                # Run optimization op (backprop)
                sess.run(classifier.optimizer,
                         feed_dict={x: batch_x, y: batch_y, teacher: batch_teacher, lr: learningRate,
                                    lambda_: 0.5})
                if i > 1: #去掉前两次数据
                    perf = time.time() - start
                    perf_list.append(perf)
                    perf_ = np.mean(perf_list)
                    fps = batchSize / perf
                    fps_list.append(fps)
                    fps_ = np.mean(fps_list)
                    if i % 10 == 0: #避免打印太多
                        print("step: {:.2f}   perf: {:.3f}  fps: {:.3f} i: {:.2f}".format(step,perf,fps_,i)) #add
            if step % displayStep == 0:
                avg_cost = 0
                avg_perf = 0
                for i in range(total_batch):
                    batch_x = batches[i]['x']
                    batch_y = batches[i]['y']
                    batch_teacher = batches[i]['teacherLogits']
                    c, p = sess.run([classifier.cost, classifier.accuracy],
                                    feed_dict={x: batch_x, y: batch_y, teacher: batch_teacher, lr: learningRate,
                                             lambda_: 0.5})
                    avg_cost += c
                    avg_perf += p
                avg_cost /= float(total_batch)
                avg_perf /= float(total_batch)
                train_loss_vals.append(avg_cost)
                train_acc_vals.append(avg_perf)
                train_iter_num.append(step)
                print("[" + get_time() + "] [Iter " + str(step) + "] Training Loss: " + \
                      "{:.6f}".format(avg_cost) + " Training Accuracy: " + "{:.5f}".format(avg_perf))
                fp.write("[" + get_time() + "] [Iter " + str(step) + "] Training Loss: " + \
                         "{:.6f}".format(avg_cost) + " Training Accuracy: " + "{:.5f}".format(avg_perf) + "\n")
                fp.flush()
                if avg_cost < -1:
                    break
            if step % testStep == 0:
                avg_cost = 0
                fin_accuracy = 0
                for i in range(total_test_batch):
                    testX = test_batches[i]['x']
                    testY = test_batches[i]['y']
                    batch_teacher = test_batches[i]['teacherLogits']
                    c, f = sess.run([classifier.cost, classifier.accuracy],
                                    feed_dict={x: testX, y: testY, teacher: batch_teacher, lr: learningRate,
                                               lambda_: 0.0})
                    avg_cost += c
                    fin_accuracy += f
                avg_cost /= float(total_test_batch)
                fin_accuracy /= float(total_test_batch)
                test_loss_vals.append(avg_cost)
                test_acc_vals.append(fin_accuracy)
                test_iter_num.append(step)
                print("[" + get_time() + "] [Iter " + str(step) + "] Testing Loss: " + \
                      "{:.6f}".format(avg_cost) + " Testing Accuracy: " + "{:.5f}".format(fin_accuracy))
                fp.write("[" + get_time() + "] [Iter " + str(step) + "] Testing Loss: " + \
                         "{:.6f}".format(avg_cost) + " Testing Accuracy: " + "{:.5f}".format(fin_accuracy) + "\n")
                fp.flush()
            if step % stepSize == 0:
                learningRate /= 10
            step += 1
        model_saver.save(sess, outputModelName)
        print("[" + get_time() + "] [Iter " + str(step) + "] Weights & Biases are saved.")

    # Print final accuracy independent of the mode
    print("[" + get_time() + "] Testing Accuracy: " + str(fin_accuracy))
    print("[" + get_time() + "] Training for val[" + str(testSet) + "] is completed.")

    # Starting building the learning curve graph
    fig, ax1 = plt.subplots()

    # Plotting training and test losses
    train_loss, = ax1.plot(train_iter_num, train_loss_vals, color='red', alpha=.5)
    test_loss, = ax1.plot(test_iter_num, test_loss_vals, linewidth=2, color='green')
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.tick_params(labelsize=15)

    # Plotting test accuracy
    ax2 = ax1.twinx()
    test_accuracy, = ax2.plot(test_iter_num, test_acc_vals, linewidth=2, color='blue')
    train_accuracy, = ax2.plot(train_iter_num, train_acc_vals, linewidth=1, color='orange')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.tick_params(labelsize=15)

    # Adding legend
    plt.legend([train_loss, test_loss, test_accuracy, train_accuracy],
               ['Training Loss', 'Val Loss', 'Val Accuracy', 'Training Accuracy'], bbox_to_anchor=(1, 0.8))
    plt.title('Learning Curve', fontsize=18)

    # Saving learning curve
    # plt.savefig(outputGraphName)
