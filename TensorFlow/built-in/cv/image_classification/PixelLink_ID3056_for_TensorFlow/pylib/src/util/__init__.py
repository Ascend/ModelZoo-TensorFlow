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
from . import log
from . import dtype
#import plt
from . import np
from . import img
_img = img
from . import dec
from . import rand
from . import mod
from . import proc
from . import test
from . import neighbour as nb
#import mask
from . import str_ as str
import io as sys_io
from . import io_ as io
from . import feature
from . import thread_ as thread
from . import caffe_ as caffe
from . import tf
from . import cmd
from . import ml
import sys
from . import url
from . import time_ as time
#from progress_bar import ProgressBar
from . import progress_bar
# log.init_logger('~/temp/log/log_' + get_date_str() + '.log')

def exit(code = 0):
    sys.exit(0)
    
is_main = mod.is_main
init_logger = log.init_logger

def get_temp_path(name = ''):
    _count = get_count();
    path = '~/temp/no-use/images/%s_%d_%s.png'%(log.get_date_str(), _count, name)
    return path
def sit(img = None, format = 'rgb', path = None, name = ""):
    if path is None:
        path = get_temp_path(name)
        
    #if img is None:
        #plt.save_image(path)
        #return path
    
        
    if format == 'bgr':
        img = _img.bgr2rgb(img)
    #if type(img) == list:
    #    plt.show_images(images = img, path = path, show = False, axis_off = True, save = True)
    #else:
    #    plt.imwrite(path, img)
    
    return path
_count = 0;

def get_count():
    global _count;
    _count += 1;
    return _count    

def cit(img, path = None, rgb = True, name = ""):
    _count = get_count();
    if path is None:
        img = np.np.asarray(img, dtype = np.np.uint8)
        path = '~/temp/no-use/images/%s_%s_%d.jpg'%(name, log.get_date_str(), _count)
        _img.imwrite(path, img, rgb = rgb)
    return path        

argv = sys.argv
    

