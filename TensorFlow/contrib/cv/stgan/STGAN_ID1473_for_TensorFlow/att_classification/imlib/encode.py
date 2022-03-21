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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

from imlib.dtype import *
from PIL import Image


def imencode(image, format='PNG', quality=95):
    """Encode an [-1.0, 1.0] into byte str.

    Args:
        format: 'PNG' or 'JPEG'.
        quality: for 'JPEG'.

    Returns:
        Byte string.
    """
    byte_io = io.BytesIO()
    image = Image.fromarray(im2uint(image))
    image.save(byte_io, format=format, quality=quality)
    bytes = byte_io.getvalue()
    return bytes


def imdecode(bytes):
    """Decode byte str to image in [-1.0, 1.0] of float64.

    Args:
        bytes: Byte string.

    Returns:
        A float64 image in [-1.0, 1.0].
    """
    byte_io = io.BytesIO()
    byte_io.write(bytes)
    image = np.array(Image.open(byte_io))
    image = uint2im(image)
    return image
