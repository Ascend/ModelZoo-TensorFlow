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
import traceback

import cv2
import numpy as np

from core.joblib import SubprocessGenerator, ThisThreadGenerator
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)


'''
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional)random_sub_size] ,
                        ...
                      ]
'''
class SampleGeneratorImageTemporal(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, temporal_image_count, sample_process_options=SampleProcessor.Options(), output_sample_types=[], **kwargs):
        super().__init__(debug, batch_size)

        self.temporal_image_count = temporal_image_count
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types

        self.samples = SampleLoader.load (SampleType.IMAGE, samples_path)

        self.generator_samples = [ self.samples ]
        self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, 0 )] if self.debug else \
                          [iter_utils.SubprocessGenerator ( self.batch_func, 0 )]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, generator_id):
        samples = self.generator_samples[generator_id]
        samples_len = len(samples)
        if samples_len == 0:
            raise ValueError('No training data provided.')

        mult_max = 4
        samples_sub_len = samples_len - ( (self.temporal_image_count)*mult_max - (mult_max-1)  )

        if samples_sub_len <= 0:
            raise ValueError('Not enough samples to fit temporal line.')

        shuffle_idxs = []

        while True:

            batches = None
            for n_batch in range(self.batch_size):

                if len(shuffle_idxs) == 0:
                    shuffle_idxs = [ *range(samples_sub_len) ]
                    np.random.shuffle (shuffle_idxs)

                idx = shuffle_idxs.pop()

                temporal_samples = []
                mult = np.random.randint(mult_max)+1
                for i in range( self.temporal_image_count ):
                    sample = samples[ idx+i*mult ]
                    try:
                        temporal_samples += SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)[0]
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(temporal_samples)) ]

                for i in range(len(temporal_samples)):
                    batches[i].append ( temporal_samples[i] )

            yield [ np.array(batch) for batch in batches]
