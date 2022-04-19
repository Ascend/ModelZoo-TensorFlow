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

from enum import Enum

VALID_COLORSPACE = {'rgb', 'bgr', 'lab', 'yuv', 'ycrcb', 'gray3d', 'gray', 'yuv', 'y'}
VALID_MODE = {'train', 'eval', 'inference', 'freeze'}
VALID_PARADIGM = {'dni'}
VALID_DEBUG_MODE = {'zeroin', 'intermediate'}
VALID_TASK = {'vsr', 'denoise', 'face', 'hdr', 'vfi'}

# HDR
HDR_CODEC_PIX_FMT = 'gbrpf32le'
HDR_FILE_SUPPORTED_EXT = 'exr'

SDR_CODEC_PIX_FMT = 'bgr24'
SDR_FILE_SUPPORTED_EXT = 'png'

RESOURCE_FILE = r'src/resource.json'

FILE_EXT_TO_PIX_FMT = {
    HDR_FILE_SUPPORTED_EXT: HDR_CODEC_PIX_FMT,
    SDR_FILE_SUPPORTED_EXT: SDR_CODEC_PIX_FMT,
}
VALID_FILE_EXT = FILE_EXT_TO_PIX_FMT.keys()


# io backend
class IO_BACKEND:
    DISK = 'disk'
    FFMPEG = 'ffmpeg'

    @classmethod
    def CHECK_VALID(cls, io_backend):
        assert io_backend in {cls.DISK, cls.FFMPEG}, \
               f'Invalid io backend {io_backend}'
