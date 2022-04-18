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
import tensorflow as tf

from .activation import Dice
from .core import DNN, LocalActivationUnit, PredictionLayer
from .interaction import (CIN, FM, AFMLayer, BiInteractionPooling, CrossNet, CrossNetMix,
                          InnerProductLayer, InteractingLayer,
                          OutterProductLayer, FGCNNLayer, SENETLayer, BilinearInteraction,
                          FieldWiseBiInteraction, FwFMLayer, FEFMLayer)
from .normalization import LayerNormalization
from .sequence import (AttentionSequencePoolingLayer, BiasEncoding, BiLSTM,
                       KMaxPooling, SequencePoolingLayer, WeightedSequenceLayer,
                       Transformer, DynamicGRU,PositionEncoding)

from .utils import NoMask, Hash, Linear, Add, combined_dnn_input, softmax, reduce_sum

custom_objects = {'tf': tf,
                  'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'AFMLayer': AFMLayer,
                  'CrossNet': CrossNet,
                  'CrossNetMix': CrossNetMix,
                  'BiInteractionPooling': BiInteractionPooling,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                  'CIN': CIN,
                  'InteractingLayer': InteractingLayer,
                  'LayerNormalization': LayerNormalization,
                  'BiLSTM': BiLSTM,
                  'Transformer': Transformer,
                  'NoMask': NoMask,
                  'BiasEncoding': BiasEncoding,
                  'KMaxPooling': KMaxPooling,
                  'FGCNNLayer': FGCNNLayer,
                  'Hash': Hash,
                  'Linear': Linear,
                  'DynamicGRU': DynamicGRU,
                  'SENETLayer': SENETLayer,
                  'BilinearInteraction': BilinearInteraction,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  'Add': Add,
                  'FieldWiseBiInteraction': FieldWiseBiInteraction,
                  'FwFMLayer': FwFMLayer,
                  'softmax': softmax,
                  'FEFMLayer': FEFMLayer,
                  'reduce_sum': reduce_sum,
                  'PositionEncoding':PositionEncoding
                  }

