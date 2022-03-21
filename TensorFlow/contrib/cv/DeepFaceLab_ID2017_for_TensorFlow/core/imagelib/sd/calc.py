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

import numpy as np
import numpy.linalg as npla

def dist_to_edges(pts, pt, is_closed=False):
    """
    returns array of dist from pt to edge and projection pt to edges
    """
    if is_closed:
        a = pts
        b = np.concatenate( (pts[1:,:], pts[0:1,:]), axis=0 )
    else:
        a = pts[:-1,:]
        b = pts[1:,:]

    pa = pt-a
    ba = b-a
    
    div = np.einsum('ij,ij->i', ba, ba)
    div[div==0]=1
    h = np.clip( np.einsum('ij,ij->i', pa, ba) / div, 0, 1 )
    
    x = npla.norm ( pa - ba*h[...,None], axis=1 )
    
    return x, a+ba*h[...,None]
    
