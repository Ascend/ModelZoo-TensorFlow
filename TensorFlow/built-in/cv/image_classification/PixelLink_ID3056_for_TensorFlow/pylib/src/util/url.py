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
import sys
import os
from six.moves import  urllib

import util
def download(url, path):
    filename = path.split('/')[-1]
    if not util.io.exists(path):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r-----Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      path, _ = urllib.request.urlretrieve(url, path, _progress)
      print()
      statinfo = os.stat(path)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    
def mdownload(urls, paths, pool_size = 4):
    assert len(urls) == len(paths)
    pool = util.thread.ThreadPool(pool_size)
    for url, path in zip(urls, paths):
        pool.add(download, [url, path])
    pool.join()
    
    
