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


# img_ref, img_dist are two images of the same size.
from metrics.VIF import vifp_mscale
from metrics.SSIM import compute_ssim
from metrics.SD import SD
from metrics.SF import spatialF
from metrics.CC import CC 
from metrics.EN import EN
from PIL import Image 
import matplotlib.pyplot as plt
import os
import numpy as np


'''
 Load the image files form the folder
 input:
     imgDir: the direction of the folder
    imgName:the name of the folder
 output:
     data:the data of the dataset
     label:the label of the datset
'''

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
# imgDir = '/Nato_camp'
imgDir = '/Nato_camp'
viFoldName = './Test_img/'+imgDir +'/vi'
irFoldName = './Test_img/'+imgDir +'/ir'
epochnum = 11
epochbegin = epochnum
vis = os.listdir(viFoldName)
irs = os.listdir(irFoldName)
while(epochnum<=11):
    resultFoldName = './result/'+ 'epoch8'
    result = os.listdir(resultFoldName)
    imgNum = len(vis)
    for i in range(imgNum):
        vi = Image.open(viFoldName+"/"+vis[i])
        ir = Image.open(irFoldName+"/"+irs[i])
        fused = Image.open(resultFoldName+"/"+result[i])
        print(vis[i])

        label.append(vis[i])
        cc1.append((CC(fused,vi)+CC(fused,ir))/2)
        sf1.append(spatialF(fused))
        ssim1.append((compute_ssim(vi,fused)+compute_ssim(ir,fused))/2)
        sd1.append(np.float64(SD(fused)))
        en1.append(EN(fused))
        vif1.append((vifp_mscale(fused,vi)+vifp_mscale(fused,ir))/2)
    epochnum += 2
    cc.append(cc1)
    sf.append(sf1)
    ssim.append(ssim1)
    sd.append(sd1)
    en.append(en1)
    vif.append(vif1)

    plt.subplot(321)
    plt.plot(range(1,imgNum+1),en1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('EN')
    plt.xlim([1,imgNum])

    plt.subplot(322)
    plt.plot(range(1,imgNum+1),sd1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('SD')
    # plt.xlim([1,imgNum])

    plt.subplot(323)
    plt.plot(range(1,imgNum+1),ssim1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('SSIM')
    # plt.xlim([1,imgNum])

    plt.subplot(324)
    plt.plot(range(1,imgNum+1),cc1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('CC')
    # plt.xlim([1,imgNum])

    plt.subplot(325)
    plt.plot(range(1,imgNum+1),sf1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('SF')
    # plt.xlim([1,imgNum])

    plt.subplot(326)
    plt.plot(range(1,imgNum+1),vif1,'o-')
    plt.legend(range(epochbegin,epochbegin+imgNum-1)) # 显示图例
    plt.title('VIF')
    # plt.xlim([1,imgNum])

    cc.append(np.float64(cc1))
    sf.append(np.float64(sf1))
    ssim.append(np.float64(ssim1))
    sd.append(np.float64(sd1))
    en.append(np.float64(en1))
    vif.append(np.float64(vif1))
    cc1 = []
    sf1 = []
    ssim1 = []
    sd1 = []
    en1 = []
    vif1 = []
    label = []
plt.show()
cc = np.array(cc)
sf = np.array(sf)
ssim = np.array(ssim)
sd = np.array(sd)
en = np.array(en)
vif = np.array(vif)

print("CC:"+str(np.average(cc[0])))
print("SF:"+str(np.average(sf[0])))
print("SSIM:"+str(np.average(ssim[0])))
print("SD:"+str(np.average(sd[0])))
print("EN:"+str(np.average(en[0])))
print("VIF:"+str(np.average(vif[0])))
plt.savefig("./"+imgDir+".png")
