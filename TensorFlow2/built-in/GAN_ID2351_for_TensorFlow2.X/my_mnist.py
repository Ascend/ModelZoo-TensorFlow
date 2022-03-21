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
"""
浠ユ鏋勫缓涓涓垜鑷繁鐨勬暟鎹泦澶勭悊杩囩▼
灏嗗鐞嗗悗鐨勬暟鎹敤numpy淇濆瓨
鍏朵粬鏂囦欢鐩存帴璇诲彇鍗冲彲
缁欎簣鍏朵粬绋嬪簭涓涓彲閫夋帴鍙 鍗崇敤浠ヤ繚瀛樻枃浠 杩樻槸閲嶆柊璇诲彇 淇濆瓨 鐒跺悗杩斿洖璇诲彇鐨勬暟鎹泦銆
閲嶆柊璇诲彇姣旇緝鑰楁椂銆傜洿鎺ヨ鍙栦繚瀛樼殑姣旇緝蹇

璁粌闆嗗拰娴嬭瘯闆嗕竴鍚岃鍙 杩斿洖
(train_imgs,train_labels),(test_imgs,test_labels)
鎺ュ彈3涓弬鏁 1涓繀椤 2涓彲閫
1 鏄惁浠庡師鏂囦欢閲嶆柊璇诲彇淇濆瓨
    2 鏄惁褰掍竴鍖
    3 鏄惁one_hot
he data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files. 
All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header. 
鏁版嵁浠ュぇ绔牸寮忓瓨鏀捐宨ntel澶勭悊鍣ㄦ槸灏忕妯″紡锛岄渶瑕佸垏鎹€
"""
import numpy as np
import sys
import struct


mnist_path = "./datasets/"
def print_log(s):
    """
    瀵筽rint鍑芥暟鐨勪竴绉嶅彉鐩搁噸鍐 闃叉鍦ㄨ鍏朵粬鏂囦欢璋冪敤鏃 杈撳嚭涓嶅繀瑕佺殑璋冭瘯淇℃伅
    """
    if __name__ =="__main__":
        print(s)
    else:
        pass
def loadMinistImage(filename):
    binfile = open(filename,'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>4i',buffers,0)#榛樿灏忕鏍煎紡 闇瑕佸彉鎴愬ぇ绔牸寮忓垯鏀规垚<
    #涓涓猧灏辨槸32浣嶆暣鏁拌〃绀4瀛楄妭,offset=4琛ㄧず鍋忕Щ4涓瓧鑺
    if head[0]!=2051:
        print_log("Errors occured in image file")
    else: 
        print_log("Read image file succeed")
        img_num = head[1]
        img_width = head[2]
        img_height = head[3]
        bits = img_num*img_height*img_width
        #60000*28*28鐨勫瓧鑺
        bits_string=">"+str(bits)+"B"#浠ヤ竴涓瓧鑺備负鍗曚綅鐨勮繛缁粨鏋勶紝鑰宨鏄互4涓瓧鑺備负鍗曚綅
        offset = struct.calcsize('4i')  # 瀹氫綅鍒癲ata寮濮嬬殑浣嶇疆
        imgs = struct.unpack_from(bits_string, buffers, offset)
        binfile.close()
        imgs = np.reshape(imgs,[img_num,img_width,img_height])
        #鍙樻垚[60000 28 28]鐨勭煩闃
        return imgs
def loadMinistLable(filename):
    binfile = open(filename,'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>2i',buffers,0)#榛樿灏忕鏍煎紡 瑕佸彉鎴愬ぇ绔牸寮忓垯鏀规垚<
    #涓涓猧灏辨槸32浣嶆暣鏁拌〃绀4瀛楄妭,offset=4琛ㄧず鍋忕Щ4涓瓧鑺 
    if head[0]!=2049:
        print_log("Errors occured in label file")
    else:
        print_log("Read label file succeed")
        img_num = head[1]
        bits = img_num
        bits_string=">"+str(bits)+"B"#浠ヤ竴涓瓧鑺備负鍗曚綅鐨勮繛缁粨鏋勶紝鑰宨鏄互4涓瓧鑺備负鍗曚綅
        offset = struct.calcsize('2i')  # 瀹氫綅鍒癲ata寮濮嬬殑浣嶇疆
        labels = struct.unpack_from(bits_string, buffers, offset)
        binfile.close()
        labels = np.reshape(labels,[img_num])
        return labels
def load_data(get_new=True,normalization=False,one_hot=False,detype=np.float32,data_path=mnist_path):
    if get_new==True:
        train_images = loadMinistImage(data_path+"/MNIST/train-images.idx3-ubyte")
        train_labels = loadMinistLable(data_path+"/MNIST/train-labels.idx1-ubyte")
        test_images = loadMinistImage(data_path+"/MNIST/t10k-images.idx3-ubyte")
        test_labels = loadMinistLable(data_path+"/MNIST/t10k-labels.idx1-ubyte")
        np.savez(data_path+"/MNIST/mnist.npz",k1=train_images,k2=train_labels,k3=test_images,k4=test_labels)
    else:
        npzfile=np.load(data_path+'/MNIST/mnist.npz') 
        train_images = npzfile['k1']
        train_labels = npzfile['k2']
        test_images = npzfile['k3']
        test_labels = npzfile['k4']
    
    if normalization==True:
        """
        to float64
        """
        train_images = train_images/255.0
        test_images = test_images/255.0
    else:
        train_images = train_images/1.0
        test_images = test_images/1.0
    
    if one_hot==True:
        new_train_labels = np.zeros(shape=(train_labels.shape[0],10))
        for i in range(train_labels.shape[0]):
            new_train_labels[i,train_labels[i]]=1.0
        new_test_labels = np.zeros(shape=(test_labels.shape[0],10))
        for i in range(test_labels.shape[0]):
            new_test_labels[i,train_labels[i]]=1.0
        return (train_images.astype(detype),new_train_labels.astype(detype)),(test_images.astype(detype),new_test_labels.astype(detype))
    else:
        return (train_images.astype(detype),train_labels.astype(detype)),(test_images.astype(detype),test_labels.astype(detype))
if __name__ == "__main__":
    (train_images,train_labels),(test_images,test_labels)=load_data(get_new=False,normalization=True,one_hot=True,detype=np.float64)
    print(train_images.shape,train_images.dtype)# (60000, 28, 28) float64
    print(train_labels.shape,train_labels.dtype)#
    print(test_images.shape,test_images.dtype)#
    print(test_labels.shape,test_labels.dtype)#
    (train_images,train_labels),(test_images,test_labels)=load_data(get_new=False,normalization=False,one_hot=False,detype=np.float32)
    print(train_images.shape,train_images.dtype)# (60000, 28, 28) float64
    print(train_labels.shape,train_labels.dtype)#
    print(test_images.shape,test_images.dtype)#
    print(test_labels.shape,test_labels.dtype)#
