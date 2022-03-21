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
from matplotlib import pyplot as plt
from scipy.integrate import simps
from matplotlib import pyplot as plt


def LandmarkError(gtLandmarks, resLandmarks, normalization='centers'):
    if normalization == 'centers':  # inter-pupil normalization
        normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
    elif normalization == 'corners':  # inter-ocular normalization
        normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
    elif normalization == 'diagonal':  # bounding box diagonal normalization
        height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
        normDist = np.sqrt(width ** 2 + height ** 2)

    error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist

    return error


def AUCError(errors, failureThreshold=0.08, step=0.0001):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
