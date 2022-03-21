# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# import os
# os.system("pip install nltk")
from npu_bridge.npu_init import *
from nltk.translate import bleu_score


class Bleu(object):
    def __init__(self):
        self.smooth_fun = bleu_score.SmoothingFunction()

    def tokenize(self, string):
        """ Specific tokenzation method need to be defined """
        raise NotImplementedError

    def score(self, references, candidate):
        """
            hypothesis: string from model output
            references: a list of strings as ground truth
        """
        # Weights are weights of 1-gram, 2-gram, 3-gram and 4-gram
        return bleu_score.sentence_bleu(references, candidate,
                                        smoothing_function=self.smooth_fun.method2)
