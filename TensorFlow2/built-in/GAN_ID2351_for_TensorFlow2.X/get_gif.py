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
#import imageio
import os
import sys

def readDirFile(path):
    """
    浠庣粰瀹氱洰褰曚腑鑾峰彇dicom鏂囦欢锛.dcm锛夌殑璇︾粏璺緞锛屽弬鏁颁负鎸囧畾鐩綍鍚嶏紝杩斿洖璺緞缁勬垚鐨勫垪琛
    """
    fileLists = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                fileLists.append(os.path.join(dirName,filename))
    return fileLists

def create_gif(source, name, duration):
	"""
     鐢熸垚gif鐨勫嚱鏁帮紝鍘熷鍥剧墖浠呮敮鎸乸ng
     source: 涓簆ng鍥剧墖鍒楄〃锛堟帓濂藉簭锛
     name 锛氱敓鎴愮殑鏂囦欢鍚嶇О
     duration: 姣忓紶鍥剧墖涔嬮棿鐨勬椂闂撮棿闅
	"""
	frames = []     # 璇诲叆缂撳啿鍖
	for img in source:
		frames.append(imageio.imread(img))#宸ヤ綔鐩綍宸茬粡杩涘叆鏂囦欢澶 鎵浠ユ瘡娆￠兘鍙皢鎵鏈夊浘鐗囨斁鍏ョ紦鍐插尯
	imageio.mimsave(name, frames, 'GIF', duration=duration) #鍒跺浘
	print("澶勭悊瀹屾垚")

os.chdir("./DCGAN_WGP_C/") #灏嗗綋鍓嶇洰褰曡皟鏁村埌鍥剧墖鏂囦欢澶逛腑
pic_list = readDirFile("./")
# os.listdir()#灏嗘墍鏈夊浘鐗囧悕璁板綍
gif_name = "C_WGPDC_gan_result.gif" # 鐢熸垚gif鏂囦欢鐨勫悕绉
duration_time = 0.1 #杩欓噷浼间箮鏈夋渶鐭棿闅 鍐嶅皬灏辨病鏈夌敤浜 鍜0.1 0.01 闂撮殧浼间箮鏄竴鏍风殑
# 鐢熸垚gif
create_gif(pic_list, gif_name, duration_time)