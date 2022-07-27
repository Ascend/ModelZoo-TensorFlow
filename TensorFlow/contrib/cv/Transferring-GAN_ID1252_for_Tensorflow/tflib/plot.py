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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	prints = []

	for name, vals in list(_since_last_flush.items()):
		prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
		_since_beginning[name].update(vals)

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(name.replace(' ', '_')+'.png')

	print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)