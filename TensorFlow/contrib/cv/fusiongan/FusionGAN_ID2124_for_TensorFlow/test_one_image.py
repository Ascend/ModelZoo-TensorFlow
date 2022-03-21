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


# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import glob
import cv2
import imageio
import os
import sys
from metrics.VIF import vifp_mscale
from metrics.SSIM import compute_ssim
from metrics.SD import SD
from metrics.SF import spatialF
from metrics.CC import CC
from metrics.EN import EN
from PIL import Image
import matplotlib.pyplot as plt
from npu_bridge.npu_init import *

dataset_path = sys.argv[1]
result_path = sys.argv[2]
print(dataset_path)
print(result_path)

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.float)
    # if is_grayscale:
    # flatten=True 以灰度图的形式读取
    #   return imageio.imread(path,  pilmode='YCbCr',as_gray=True).astype(np.float)
    # else:
    #   return imageio.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
    return imageio.imwrite(path, image)
  
  
def prepare_data(dataset):
    data_dir = dataset
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.compat.v1.variable_scope('fusion_model'):
        with tf.compat.v1.variable_scope('layer1'):
            weights=tf.compat.v1.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.compat.v1.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir= tf.nn.conv2d(conv4_ir, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    


def input_setup(index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi


num_epoch=11
while(num_epoch<=11):
    print(num_epoch)

    reader = tf.compat.v1.train.NewCheckpointReader('./test/checkpoint/CGAN_120/CGAN.model-'+ str(num_epoch))  #add 

    with tf.compat.v1.name_scope('IR_input'):
        #红外图像patch
        images_ir = tf.compat.v1.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.compat.v1.name_scope('VI_input'):
        #可见光图像patch
        images_vi = tf.compat.v1.placeholder(tf.float32, [1,None,None,None], name='images_vi')
        #self.labels_vi_gradient=gradient(self.labels_vi)
        #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
        input_image=tf.concat([images_ir,images_vi],axis=-1)
    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image)


    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        foldername = 'Nato_camp'
        # foldername = 'else'
        data_ir=prepare_data(dataset_path + '/Test_img/'+foldername + '/ir')
        data_vi=prepare_data(dataset_path + '/Test_img/'+foldername + '/vi')
        for i in range(len(data_ir)):
            start=time.time()
            train_data_ir, train_data_vi=input_setup(i)
            result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            result=result*127.5+127.5
            result = result.squeeze()
            image_path = os.path.join(result_path, 'result', foldername,'epoch'+str(num_epoch))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if i+1<=9:
                image_path = os.path.join(image_path,'0'+str(i+1)+".bmp")
            else:
                image_path = os.path.join(image_path,''+str(i+1)+".bmp")
            end=time.time()
            # print(out.shape)
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
    tf.reset_default_graph()
    num_epoch=num_epoch+1



cc = []
sf = []
ssim = []
sd = []
en = []
vif = []
label = []
cc1 = []
sf1 = []
ssim1 = []
sd1 = []
en1 = []
vif1 = []
label = []

imgDir = 'Nato_camp'
# imgDir = 'else'
viFoldName = os.path.join(dataset_path, 'Test_img', imgDir ,'vi')
irFoldName = os.path.join(dataset_path, 'Test_img', imgDir ,'ir')
epochnum = 11
epochbegin = epochnum
vis = os.listdir(viFoldName)
irs = os.listdir(irFoldName)
while(epochnum<=11):
    resultFoldName = os.path.join(result_path, 'result',imgDir,'epoch'+str(epochnum))
    result = os.listdir(resultFoldName)
    print('epoch:'+str(epochnum))
    print(irs)
    # print(vis)
    # print(result)
    # print(len(vis))
    imgNum = len(vis)
    for i in range(imgNum):
        vi = Image.open(viFoldName+"/"+vis[i])
        ir = Image.open(irFoldName+"/"+irs[i])
        fused = Image.open(resultFoldName+"/"+result[i])
        print(vis[i])
        # print(np.array(vi.shape))
        # print(np.array(ir.shape))
        # print(np.array(fused.shape))
        # plt.imshow(np.array(fused))
        label.append(vis[i])
        cc1.append((CC(fused,vi)+CC(fused,ir))/2)
        sf1.append(spatialF(fused))
        ssim1.append((compute_ssim(vi,fused)+compute_ssim(ir,fused))/2)
        sd1.append(np.float64(SD(fused)))
        en1.append(EN(fused))
        vif1.append((vifp_mscale(fused,vi)+vifp_mscale(fused,ir))/2)
    epochnum += 1
    cc.append(cc1)
    sf.append(sf1)
    ssim.append(ssim1)
    sd.append(sd1)
    en.append(en1)
    vif.append(vif1)

    plt.subplot(321)
    plt.plot(range(1,imgNum+1),en1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) 
    plt.title('EN')
    plt.xlim([1,imgNum])

    plt.subplot(322)
    plt.plot(range(1,imgNum+1),sd1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1))
    plt.title('SD')
    # plt.xlim([1,imgNum])

    plt.subplot(323)
    plt.plot(range(1,imgNum+1),ssim1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1))
    plt.title('SSIM')
    # plt.xlim([1,imgNum])

    plt.subplot(324)
    plt.plot(range(1,imgNum+1),cc1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1))
    plt.title('CC')
    # plt.xlim([1,imgNum])

    plt.subplot(325)
    plt.plot(range(1,imgNum+1),sf1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) 
    plt.title('SF')
    # plt.xlim([1,imgNum])

    plt.subplot(326)
    plt.plot(range(1,imgNum+1),vif1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) 
    plt.title('VIF')
    # plt.xlim([1,imgNum])

    cc.append(cc1)
    sf.append(sf1)
    ssim.append(ssim1)
    sd.append(sd1)
    en.append(en1)
    vif.append(vif1)
    cc1 = []
    sf1 = []
    ssim1 = []
    sd1 = []
    en1 = []
    vif1 = []
    label = []
plt.show()

plt.savefig(result_path+'/'+imgDir+".png")


print(label)
print("cc: ",np.mean(cc[0]))
print("sf: ",np.mean(sf[0]))
print("ssim: ",np.mean(ssim[0]))
#print("sd",np.mean(sd[0]))
print("en: ",np.mean(en[0]))
print("vif: ",np.mean(vif[0]))
print("all_result =  cc: %4.4f sf: %4.4f ssim: %4.4f  en: %4.4f vif: %.8f" % (np.mean(cc[0]),np.mean(sf[0]),np.mean(ssim[0]),np.mean(en[0]),np.mean(vif[0])))
