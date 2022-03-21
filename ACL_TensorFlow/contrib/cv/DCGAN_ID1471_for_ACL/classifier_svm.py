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
from sklearn import svm
import os
import tensorflow as tf
import numpy as np
import main
from model import DCGAN
import joblib
from six.moves import xrange
from utils import *
from glob import glob

# python3 classifier_svm.py --dataroot=/media/annusha/BigPapa/Study/DL/food-101/images --imageSize=32
# --dcgan=/media/annusha/BigPapa/Study/DL/out_imagenet/netD_epoch_7.pth
FLAGS = main.FLAGS

if __name__ == '__main__':

    fname = 'cifar10_svm'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if not FLAGS.train_svm:
            print('save features')

            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                z_dim=FLAGS.z_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir,
                out_dir=FLAGS.out_dir,
                max_to_keep=FLAGS.max_to_keep)

            # 加载数据集
            data_dir = './data/cifar10/'
            file_dir = os.listdir(data_dir)
            num = 10000
            label_datas = []
            image_datas = []
            for i in range(0, len(file_dir)):
                path = os.path.join(data_dir, file_dir[i])

                if str(file_dir[i])[-3:] == 'bin' and os.path.isfile(path):
                    bytestream = open(path, "rb")
                    buf = bytestream.read(num * (1 + 32 * 32 * 3))
                    bytestream.close()

                    data = np.frombuffer(buf, dtype=np.uint8)
                    data = data.reshape(num, 1 + 32 * 32 * 3)
                    labels_images = np.hsplit(data, [1])
                    label_data = labels_images[0].reshape(num)
                    label_datas.extend(label_data)
                    image_data = labels_images[1].reshape(num, 32, 32, 3)
                    for ni in range(num):
                        img = np.reshape(image_data[ni], (3, 32, 32))
                        img = img.transpose(1, 2, 0)
                        image_datas.append(img)

            batch_images = []
            # origin_data = glob(os.path.join("./data/cifar10/materials", '*.jpg'))
            batch_idxs = min(len(image_datas), FLAGS.train_size)
            print("batch index:", batch_idxs)
            for idx in xrange(0, int(batch_idxs)):
                batch_files = image_datas[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
                batch = [
                    transform(img,
                              input_height=FLAGS.input_height,
                              input_width=FLAGS.input_width,
                              resize_height=FLAGS.output_height,
                              resize_width=FLAGS.output_width,
                              crop=FLAGS.crop) for img in batch_files]
                batch_image = np.array(batch).astype(np.float32)
                batch_images.append(batch_image)
            print("** data loaded **")

            # 载入meta参数
            ck_dir = "./out/test/checkpoint"
            load_success, load_counter = dcgan.load(ck_dir)
            if not load_success:
                raise Exception("Checkpoint not found in " + ck_dir)

            features = np.array([])
            labels = np.array([])
            i = 0

            print("batch images length: ", len(batch_images))
            label_datas = np.array(label_datas)
            print("label datas shape", label_datas.shape)
            for img in batch_images:
                print("get feature start...")
                try:
                    input_v = tf.cast(tf.convert_to_tensor(img), tf.float32)
                except Exception as e:
                    print(e, "test")
                feature = dcgan.discriminator(input_v, reuse=True, feature=True)
                feature = feature.eval(session=sess)
                feature = feature.astype(np.float16)
                label = label_datas[i*feature.shape[0]:(i+1)*feature.shape[0], ]
                if i == 0:
                    features = feature
                    labels = label
                else:
                    features = np.concatenate((features, feature), axis=0)
                    labels = np.concatenate((labels, label), axis=0)
                if i % 5 == 0 and i:
                    print('processed ', i)
                    break
                if i == 50:
                    break
                i = i + 1

            # split to train and validation sets
            # indexes = list(range(len(labels)))
            # print('number of samples ', labels.shape)
            print(features.shape)
            print(labels.shape)

            features = np.concatenate((features, labels[:, np.newaxis]), axis=1)

            features = features.astype(np.float16)

            np.savetxt(fname, features)

        else:
            # if load features from file
            print('load features')
            data = np.loadtxt(fname, dtype=np.float16)
            features, labels = data[:, : -1], data[:, -1:]

            print('number of samples ', labels.shape)
            print(features.shape)

            indexes = list(range(len(labels)))
            np.random.shuffle(indexes)

            ratio = int(0.5 * len(labels))

            train_data, train_labels = features[indexes[: ratio]], labels[indexes[: ratio]]
            val_data, val_labels = features[indexes[ratio:]], labels[indexes[ratio:]]
            print('len train :', len(train_labels))
            print('len val: ', len(val_labels))

            print('train svm')
            # clf = svm.SVC(decision_function_shape='ovo')
            # clf = svm.LinearSVC(penalty='l2')
            # clf.fit(train_data, train_labels)
            # joblib.dump(clf, 'svm.pkl')

            print('download svm')
            clf = joblib.load('svm.pkl')

            print('predict svm')
            val_labels = val_labels.squeeze()
            predicted_labels = clf.predict(val_data)
            print(len(val_labels), len(predicted_labels))
            a = predicted_labels == val_labels
            print(np.sum(a))
            accuracy = np.sum(predicted_labels == val_labels) / len(val_labels)

            print('svm results: %.4f accuracy' % accuracy)

            # precision and recall
            uniq_labels = set(val_labels)
            for label in uniq_labels:
                tp = np.sum(predicted_labels[predicted_labels == label] ==
                            val_labels[predicted_labels == label])
                fn_tp = np.sum(val_labels == label)
                fn = fn_tp - tp
                fp = np.sum(predicted_labels[predicted_labels == label] !=
                            val_labels[predicted_labels == label])

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                print('class %d :: precision %.4f  recall %.4f ' % (label, precision, recall))
