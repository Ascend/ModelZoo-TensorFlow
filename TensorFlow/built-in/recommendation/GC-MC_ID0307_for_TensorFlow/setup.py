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
from setuptools import setup
from setuptools import find_packages

setup(name='gcmc',
      version='0.1',
      description='Graph Convolutional Matrix Completion',
      author='Rianne van den Berg, Thomas Kipf',
      author_email='riannevdberg@gmail.com',
      url='http://riannevdberg.github.io',
      download_url='https://github.com/riannevdberg/gc-mc',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'scipy',
                        'pandas',
                        'h5py'
                        ],
      package_data={'gcmc': ['README.md']},
      packages=find_packages())
