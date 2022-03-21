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


from npu_bridge.npu_init import *
from scipy.io import wavfile
from pesq import pesq
from pystoi.stoi import stoi
from glob import glob
import os, librosa, sys, argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-r', "--results_root", help="Path to test results folder", required=True)
args = parser.parse_args()

sr = 16000

all_files = glob("{}/wavs/*.wav".format(args.results_root))
gt_folder = args.results_root + '/gts/{}'

print('Calculating for {} files'.format(len(all_files)))

total_pesq = 0
total_stoi = 0
total_estoi = 0

for filename in tqdm(all_files):
	gt_filename = gt_folder.format(os.path.basename(filename))
	rate, deg = wavfile.read(filename)
	rate, ref = wavfile.read(gt_filename)
	if len(ref.shape) > 1: ref = np.mean(ref, axis=1) #raise ValueError('Audio should be a mono band')

	if rate != sr:
		ref = librosa.resample(ref.astype(np.float32), rate, sr).astype(np.int16)
		rate = sr

	if len(ref) > len(deg): x = ref[0 : deg.shape[0]]
	elif len(deg) > len(ref):
		deg = deg[: ref.shape[0]]
		x = ref
	else: x = ref

	total_pesq += pesq(rate, x, deg, 'nb')
	total_stoi += stoi(x, deg, rate, extended=False)
	total_estoi += stoi(x, deg, rate, extended=True)

print('Mean PESQ: {}'.format(total_pesq / len(all_files)))
print('Mean STOI: {}'.format(total_stoi / len(all_files)))
print('Mean ESTOI: {}'.format(total_estoi / len(all_files)))

