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
from .afm import AFM
from .autoint import AutoInt
from .ccpm import CCPM
from .dcn import DCN
from .dcnmix import DCNMix
from .deepfefm import DeepFEFM
from .deepfm import DeepFM
from .difm import DIFM
from .fgcnn import FGCNN
from .fibinet import FiBiNET
from .flen import FLEN
from .fnn import FNN
from .fwfm import FwFM
from .ifm import IFM
from .mlr import MLR
from .multitask import SharedBottom, ESMM, MMOE, PLE
from .nfm import NFM
from .onn import ONN
from .pnn import PNN
from .sequence import DIN, DIEN, DSIN, BST
from .wdl import WDL
from .xdeepfm import xDeepFM

__all__ = ["AFM", "CCPM", "DCN", "IFM", "DIFM", "DCNMix", "MLR", "DeepFM", "MLR", "NFM", "DIN", "DIEN", "FNN", "PNN",
           "WDL", "xDeepFM", "AutoInt", "ONN", "FGCNN", "DSIN", "FiBiNET", 'FLEN', "FwFM", "BST", "DeepFEFM",
           "SharedBottom", "ESMM", "MMOE", "PLE"]

