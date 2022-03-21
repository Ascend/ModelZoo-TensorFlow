# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
"""Calculate the difference of output node output value when using PB model inference and OM model inference
An example of command-line usage is:
python3 cal_difference.py offline_inference/pb_output offline_inference/om_output
"""
import os
import sys
import numpy as np


def main(argv):
    npz_filename_list = []
    om_output_list = []
    for root, dirs, files in os.walk(argv[0]):
        for f in files:
            npz_filename_list.append(f)
        npz_root = root
    for root, dirs, files in os.walk(argv[1]):
        for f in files:
            om_output_list.append(f)
        om_output_root = root

    fenshu = []
    all_fenshu = 0
    for index in range(len(npz_filename_list)):
        npz_file = npz_filename_list[index]
        om_output_file = om_output_list[index]
        full_npz_file = os.path.join(npz_root, npz_file)
        full_om_output_file = os.path.join(om_output_root, om_output_file)
        print("Start to calculate the difference between %s and %s" % (full_npz_file, full_om_output_file))
        pb_outfile = np.load(full_npz_file)
        pb_out = pb_outfile['arr_0']
        pb_out1 = np.reshape(pb_out, (68644, 64))
        om_out = np.genfromtxt(full_om_output_file)

        temp_fenshu1 = []
        for i in range(68644):
            for j in range(64):
                all_fenshu += abs(om_out[i][j] - pb_out1[i][j])
                if pb_out1[i][j] != 0:
                    temp_fenshu1.append(abs(om_out[i][j] - pb_out1[i][j]) / pb_out1[i][j])
        fenshu.append(np.mean(temp_fenshu1))
        print("Calculation between %s and %s ended" % (full_npz_file, full_om_output_file))
    print("The percentage of the average difference relative to the pb model inference output is %f" % (np.mean(fenshu)))


if __name__ == '__main__':
    main(sys.argv[1:])
