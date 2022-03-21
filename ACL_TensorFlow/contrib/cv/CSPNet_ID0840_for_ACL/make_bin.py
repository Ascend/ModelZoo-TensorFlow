# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pickle

def onehot(index):
	""" It creates a one-hot vector with a 1 in
		position represented by index
	"""
	onehot = np.zeros(10)
	onehot[index] = 1
	return onehot

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		X = dict[b'data']
		Y = dict[b'labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
		Y = np.array(Y)
		return X,Y

cifar_path = './data'
Xte,Yte = unpickle(os.path.join(cifar_path,'test_batch'))
Xte.tofile('./data/Xtest.bin')
Yte = np.array([onehot(x) for x in Yte]).astype('int64')
Yte.tofile('./data/Ytest.bin')

