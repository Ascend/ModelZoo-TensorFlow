#
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
#
import os
import soundfile as sf
import librosa
import sys

# root_dir = "/fast/datalhf/vctk/VCTK-Corpus/wav8"
# target_sr = 8000

def resample_file(father_dir, target_sr):
  print('get in',father_dir)
  file_or_fold_list = os.listdir(father_dir)
  for file_or_fold in file_or_fold_list:
    file_or_fold_dir = os.path.join(father_dir,file_or_fold)
    if os.path.isdir(file_or_fold_dir):
      resample_file(file_or_fold_dir, target_sr)
    elif file_or_fold_dir[-4:] == '.wav':
      data, sr = sf.read(file_or_fold_dir)
      if sr != target_sr:
        data = librosa.resample(data, sr, target_sr, res_type='kaiser_fast')
        print('resample wav(%d to %d) :' % (sr, target_sr), file_or_fold_dir)
        sf.write(file_or_fold_dir, data, target_sr)

if __name__ == '__main__':
  assert len(sys.argv) == 3, 'argv error.'
  root_dir = sys.argv[1]
  sr = int(sys.argv[2])
  resample_file(root_dir, sr)
