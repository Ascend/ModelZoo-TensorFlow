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


class SampleGeneratorImage(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, sample_process_options=SampleProcessor.Options(), output_sample_types=[], raise_on_no_data=True, **kwargs):
        super().__init__(debug, batch_size)
        self.initialized = False
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types

        samples = SampleLoader.load (SampleType.IMAGE, samples_path)
        
        if len(samples) == 0:
            if raise_on_no_data:
                raise ValueError('No training data provided.')
            return
        
        self.generators = [ThisThreadGenerator ( self.batch_func, samples )] if self.debug else \
                          [SubprocessGenerator ( self.batch_func, samples )]

        self.generator_counter = -1
        self.initialized = True
        
    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, samples):
        samples_len = len(samples)
        

        idxs = [ *range(samples_len) ]
        shuffle_idxs = []

        while True:

            batches = None
            for n_batch in range(self.batch_size):

                if len(shuffle_idxs) == 0:
                    shuffle_idxs = idxs.copy()
                    np.random.shuffle (shuffle_idxs)

                idx = shuffle_idxs.pop()
                sample = samples[idx]
                
                x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]

                for i in range(len(x)):
                    batches[i].append ( x[i] )

            yield [ np.array(batch) for batch in batches]
