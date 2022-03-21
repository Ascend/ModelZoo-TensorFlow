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
from npu_bridge.npu_init import *
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy



ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = [ "-std=c++11", "-fopenmp",],
       extra_link_args=["-std=c++11", '-fopenmp'],
  )]

setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)

