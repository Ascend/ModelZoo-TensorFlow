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

import numpy as np
import tensorflow as tf
import argparse
import time
import os

from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split
from devnet import *


tf.set_random_seed(42)
np.random.seed(42)
MAX_INT = np.iinfo(np.int32).max
data_format = 0


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """
    batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size,
                                                                     rng)
        counter += 1
        yield (ref, training_labels)
        if (counter > nb_batch):
            counter = 0


def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    """
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    """
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if (i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)


def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    """
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    """
    ref = np.empty((batch_size))
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if (i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, network_depth, x_test):
    """
    load the saved weights to make predictions
    """
    scoring_network = DeviationNetwork(x_test.shape, network_depth, learning_rate=0.001)
    scoring_network.load_weights(model_name)

    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def inject_noise_sparse(seed, n_out, random_seed):
    """
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    """
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()


def inject_noise(seed, n_out, random_seed):
    """
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    """
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

def eval_devnet(args):
    """
    eval devnet
    """
    names = ['annthyroid_21feat_normalised']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        filename = nm.strip()
        x, labels = dataLoading(args.input_path + filename + ".csv")
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                            stratify=labels)
        y_test = np.array(y_test)
        x_test = x_test.astype(np.float32)
        model_name = "./model/devnet_annthyroid_21feat_normalised_0.02cr_512bs_30ko_4d"
        scores = load_model_weight_predict(model_name, network_depth, x_test)
        print("[INFO] output shape: "+ str(scores.shape))
        print("[INFO] label type: "+ str(y_test.dtype))
        print("[INFO] data type: "+str(x_test.dtype))
        rauc, ap = aucPerformance(scores, y_test)
        x_test.tofile("test_input.bin")
        y_test.tofile("test_label.bin")


def run_devnet(args):
    """
    Training DevNet
    """
    names = ['annthyroid_21feat_normalised']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path +"/"+filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path +"/"+filename + ".svm")
            x = x.tocsr()
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                stratify=labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            n_noise = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(random_seed)
            if data_format == 0:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)

                noises = inject_noise(outliers, n_noise, random_seed)
                x_train = np.append(x_train, noises, axis=0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]

                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            input_shape = x_train.shape[1:]
            n_samples_trn = x_train.shape[0]
            n_outliers = len(outlier_indices)
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            start_time = time.time()
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size
            batch_input_shape = list((batch_size,) + tuple(input_shape))
            batch_label_shape = [batch_size, 1]
            nb_batch = args.nb_batch
            model = DeviationNetwork(batch_input_shape, network_depth, label_shape=batch_label_shape,
                                    learning_rate=0.001, use_npu=False)
            model_name = "./model/devnet_" + filename + "_" + str(args.cont_rate) + "cr_" + str(
                args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) + "d"

            model.fit_generator(
                batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                steps_per_epoch=nb_batch,
                epochs=epochs,
                model_path=model_name)
            train_time += time.time() - start_time

            start_time = time.time()
            scores = load_model_weight_predict(model_name, network_depth, x_test)
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time / runs
        test_time = test_time / runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename + '_' + str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org,n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1', '2', '4'], default='4',
                    help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=3,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--data_set", type=str, default='annthyroid_21feat_normalised', help="a list of data set names")
parser.add_argument("--data_format", choices=['0', '1'], default='0',
                    help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str,
                    default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv',
                    help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
parser.add_argument("--data_url", type=str, default='', help="data_url")
parser.add_argument("--train_url", type=str, default='', help="train_url")
args = parser.parse_args()

# this code is uesd for pycharm-toolkit and modelarts
# import moxing as mox
# os.mkdir('./model/')
# os.mkdir(os.path.dirname(args.output))
# mox.file.copy_parallel(args.data_url, args.input_path)
# run_devnet(args)
# mox.file.copy_parallel(os.path.dirname(args.output), args.train_url)
# mox.file.copy_parallel('./model/', args.train_url)

args = parser.parse_args()
run_devnet(args)
