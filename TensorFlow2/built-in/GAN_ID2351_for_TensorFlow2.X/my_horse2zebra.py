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
#from PIL import Image
import numpy as np
import os

def readDirFile(path):
    fileLists = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if ".jpg" in filename.lower():  # check whether the file's DICOM
                fileLists.append(os.path.join(dirName,filename))
    return fileLists
def img2numpy(path):
    file_list = readDirFile(path)
    buf=[]
    for path in file_list:
        temp_im = Image.open(path).convert("RGB")#鎭掑畾涓轰笁閫氶亾 鍥犱负trainB涓湁寮傚父鍥剧嚎闇瑕佸仛娆″鐞
        buf.append(np.array(temp_im))
    buf=np.stack(buf,axis=0)
    # axis鎸囧畾鏂拌酱鍦ㄧ粨鏋滃昂瀵镐腑鐨勭储寮 褰揳xis=0 鍗砨atch缁村害
    # 鎬诲厓绱犱釜鏁颁负len(buf)*buf.shape[0]*buf.shape[1]...buf.shape[n]
    # 鍫嗗彔鍚巗hape涓 [... len(buf) ...] len(buf)鍦╯hape涓殑绱㈠紩鍊(浣嶇疆) 鍗砤xis鎸囧畾鐨勫
    # 杩欎簺璁＄畻瀹屾垚鍚,鎸夌収浠庡悗寰鍓嶇殑椤哄簭寮濮嬪爢鍙 閬囧埌len(buf)鎵鍦ㄧ淮搴 鍗砤xis鎸囧畾缁村害鏃 璇ョ淮搴︽墍鏈夊厓绱犳槸list涓悇鎴愬憳瀵瑰簲浣嶇疆(鍏朵綑缁村害瀛樺湪涓旂‘瀹)鐨勫崟涓鍏冪礌璐＄尞鍑烘潵鐨勯泦鍚
    return buf

def load_horse2zebra(headpath,get_new=True,detype=np.uint8):
    """
    get_new=False鍙湁鍦ㄥ凡缁忓瓨鍦╤orse2zebra.npz鏃舵湁鏁 灏变笉閫氳繃os.walk() 鍒ゅ畾浜 瓒呭嚭浜嗗鏉傚害
    normalization浼氬嚭鐜版柊鐨勫疄渚 涓嶅缓璁斁鍦ㄦ澶
    """
    if get_new == True:
        testA = img2numpy(headpath+"testA/")
        testB = img2numpy(headpath+"testB/")
        trainA = img2numpy(headpath+"trainA/")
        trainB = img2numpy(headpath+"trainB/")
        np.savez(headpath+"horse2zebra.npz",k1=testA,k2=testB,k3=trainA,k4=trainB)
    else:
        npzfile=np.load(headpath+'horse2zebra.npz') 
        testA = npzfile['k1']
        testB = npzfile['k2']
        trainA = npzfile['k3']
        trainB = npzfile['k4']
    return (testA.astype(detype),testB.astype(detype)),(trainA.astype(detype),trainB.astype(detype))
if __name__ == "__main__":
    (testA,testB),(trainA,trainB)=load_horse2zebra("./datasets/horse2zebra/",get_new=True,detype=np.uint8)
    print(testA.shape,testA.dtype)
    print(testB.shape,testB.dtype)
    print(trainA.shape,trainA.dtype)
    print(trainB.shape,trainB.dtype)

