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
import argparse
import cv2 as cv
import numpy as np
from scipy.integrate import simps


class LandmarkMetric:
    """
    The normalized average point-to-point Euclidean error, i.e., MSE normalized by a certain size
    (measured as the Euclidean distance between the outer corners of the eyes) will be used as the error measure.
    """

    def __init__(self, num_lmk=68, norm_type=None):
        """

        Args:
            num_lmk: int, number of landmarks, 68 by default.
            norm_type: one of `WITHOUT_NORM`, ` OCULAR`,  DIAGONAL`
                and `PUPIL`， which specifies the normalization factor of MSE

                    WITHOUT_NORM = 0
                    OCULAR = 1
                    PUPIL = 2
                    DIAGONAL = 3
        """

        self.num_lmk = num_lmk
        self.norm_type = norm_type

    def __call__(self, y, y_hat):
        """

        Args:
            y: np.array, [N_landmark, 2]
            y_hat: np.array, [N_Landmark, 2]

        Returns:

        """
        WITHOUT_NORM = 0  # common MSE
        # the corner inter-ocular distance fails to give a meaningful localisation metric in the case of profile
        # views as it becomes a very small value
        OCULAR = 1
        PUPIL = 2
        DIAGONAL = 3

        #  np.linalg.norm 范数，L2-norm by default
        #  矩阵做差-平方-按1轴求和再开方-按0轴平均，即：计算每个点的真值与预测值的L2norm(欧式距离)，再对所有点求平均
        avg_ptp_dis = np.mean(np.linalg.norm(y - y_hat, axis=1))
        norm_dist = 1

        if self.norm_type == OCULAR or self.norm_type == PUPIL:
            assert y.shape[0] == 68, "number of landmark must be 68"

        if self.norm_type == PUPIL:
            norm_dist = np.linalg.norm(np.mean(y[36:42], axis=0) - np.mean(y_hat[42:48], axis=0))

        elif self.norm_type == OCULAR:
            norm_dist = np.linalg.norm(y[36] - y_hat[45])

        elif self.norm_type == DIAGONAL:
            height, width = np.max(y, axis=0) - np.min(y_hat, axis=0)
            norm_dist = np.sqrt(width ** 2 + height ** 2)

        rmse = avg_ptp_dis / norm_dist
        return rmse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', default=1, type=int,
                        help='batchsize')
    parser.add_argument('--metric', default=3, type=int,
                        help="""one number of 0(WITHOUT_NORM) 1(OCULAR) 2(PUPIL) 3(DIAGONAL), which specifies the 
                        normalization factor of MSE """)
    parser.add_argument('--bin_dir', default=r'bin/testset', type=str,
                        help="""the bin data path""")
    parser.add_argument('--ptv_dir', default=r'datasets/testset', type=str,
                        help="""the ptv data path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def read_bin_file(file_path):
    bin_files = os.listdir(file_path)
    loaded_dic = {}
    for bin_file in bin_files:
        file_name = bin_file.split("_")[0]
        loaded_dic.update({file_name: bin_file})

    return loaded_dic


def load_image(image_path):
    im = cv.imread(image_path)
    return im


def AUCError(errors, failureThreshold=0.08, step=0.0001):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))


def main():
    args = parse_args()
    # data load and preprocess
    # load bin data
    print("bin images data loaded")

    # load dataset test data
    files_name = os.listdir(args.bin_dir)

    errs = []

    metric = LandmarkMetric(68, args.metric)

    for file_name in files_name:
        file_path = os.path.join(args.bin_dir, file_name)

        pred = np.fromfile(file_path, dtype=np.float32)
        pred = np.reshape(pred, (68, 2))

        orig_path = os.path.join(args.ptv_dir, file_name.split("_output_")[0] + '.ptv')
        if os.path.isfile(orig_path):
            orig = np.loadtxt(orig_path, dtype=np.float32, delimiter=',')
            test_err = metric(np.squeeze(orig), np.squeeze(pred))
            errs.append(test_err)

            print('The mean error for image {} is: {:.5f}'.format(file_name.split("_output_")[0], test_err))
        else:
            print('[WARNING]The file %s does not exist' % file_name.split("_output_")[0] + '.ptv')

    errs = np.array(errs)
    print('The overall mean error is: {:.5f}'.format(np.mean(errs)))
    AUCError(errs)


if __name__ == '__main__':
    main()
