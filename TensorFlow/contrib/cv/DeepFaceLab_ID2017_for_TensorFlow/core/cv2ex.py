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

import cv2
import numpy as np
from pathlib import Path
from core.interact import interact as io
from core import imagelib 
import traceback

def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, flags)
    except:
        if verbose:
            io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None

def cv2_imwrite(filename, img, *args):
    ret, buf = cv2.imencode( Path(filename).suffix, img, *args)
    if ret == True:
        try:
            with open(filename, "wb") as stream:
                stream.write( buf )
        except:
            pass

def cv2_resize(x, *args, **kwargs):
    h,w,c = x.shape
    x = cv2.resize(x, *args, **kwargs)
    
    x = imagelib.normalize_channels(x, c)
    return x
    