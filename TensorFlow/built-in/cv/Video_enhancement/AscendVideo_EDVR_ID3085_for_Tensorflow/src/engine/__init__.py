# Copyright 2022 Huawei Technologies Co., Ltd
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


def build_engine(cfg):
    """Returns the engine class given the mode in cfg.

    Args:
        cfg: yacs node, global configuration.
    
    Returns:
        engine class.
    """
    mode = cfg.mode
    ckpt = cfg.checkpoint
    if mode == 'train':
        from .trainer import SessionTrainer
        return SessionTrainer
    elif mode == 'inference':
        from .inferencer import SessionInferencer, ModelFreeInferencer
        if ckpt.endswith(".pb"):
            return ModelFreeInferencer
        else:
            return SessionInferencer
    elif mode == 'freeze':
        from .freezer import SessionFreezer
        return SessionFreezer
    else:
        raise NotImplementedError


__all__ = ['build_engine']
