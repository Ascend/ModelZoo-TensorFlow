# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
Static Variable in DQN
"""
TAU = 0.001  # Target Network HyperParameters
LRC = 0.001  # Lerning rate for Critic
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
GAMMA = 0.99
UPDATE_FREQ = 64
