"""LICENSE"""
from npu_bridge.npu_init import *
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
imgDir = '/else'
#viFoldName = './Test_img/'+imgDir +'/vi'
#irFoldName = './Test_img/'+imgDir +'/ir'



path = './test_imgs'
viFoldName = path + '/VIS/'
irFoldName = path + '/IR/'

epochnum = 20
epochbegin = epochnum
vis = os.listdir(viFoldName)
irs = os.listdir(irFoldName)

while(epochnum <= 20):
    #resultFoldName = './result'+imgDir+'/epoch'+str(epochnum)
    resultFoldName = './results'
    result = os.listdir(resultFoldName)
    print('epoch:' + str(epochnum))
    #print(irs)
    # print(vis)
    # print(result)
    # print(len(vis))
    imgNum = len(vis)
    for i in range(imgNum):
        print(viFoldName + 'VIS' + str(i + 1) + '.bmp')
        print(irFoldName + 'IR' + str(i + 1) + '.bmp')
        vi = Image.open(viFoldName + 'VIS' + str(i + 1) + '.bmp')
        ir = Image.open(irFoldName + 'IR' + str(i + 1) + '.bmp')
        #vi = Image.open(viFoldName+"/"+vis[i])
        #ir = Image.open(irFoldName+"/"+irs[i])
        fused = Image.open(resultFoldName + "/" + str(i + 1) + '.bmp')
        print(resultFoldName + "/" + str(i + 1) + '.bmp')
        # print(np.array(vi.shape))
        # print(np.array(ir.shape))
        # print(np.array(fused.shape))
        # plt.imshow(np.array(fused))
        label.append(vis[i])
        cc1.append((CC(fused, vi) + CC(fused, ir))/2)
        sf1.append(spatialF(fused))
        ssim1.append((compute_ssim(vi, fused) + compute_ssim(ir, fused))/2)
        sd1.append(np.float64(SD(fused)))
        en1.append(EN(fused))
        vif1.append((vifp_mscale(fused, vi) + vifp_mscale(fused, ir))/2)
    epochnum += 2
    cc.append(cc1)
    sf.append(sf1)
    ssim.append(ssim1)
    sd.append(sd1)
    en.append(en1)
    vif.append(vif1)


    CC_SUM = sum(cc1)
    SSIM_SUM = sum(sf1)
    SF_SUM = sum(ssim1)
    SD_SUM = sum(sd1)
    EN_SUM = sum(en1)
    VIF_SUM = sum(vif1)
    print(CC_SUM)
    print(SSIM_SUM)
    print(SF_SUM)
    print(SD_SUM)
    print(EN_SUM)
    print(VIF_SUM)

    print('CC: %s, SF: %s, SSIM: %s, SD: %s, EN: %s, VIF: %s' % (
        CC_SUM / 20, SSIM_SUM / 20, SF_SUM / 20, SD_SUM / 20, EN_SUM / 20,
        VIF_SUM / 20))






    plt.subplot(321)
    plt.plot(range(1, imgNum + 1), en1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
    plt.title('EN')
    plt.xlim([1, imgNum])


    plt.subplot(322)
    plt.plot(range(1, imgNum + 1), sd1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
    plt.title('SD')
    # plt.xlim([1,imgNum])

    plt.subplot(323)
    plt.plot(range(1, imgNum + 1), ssim1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
    plt.title('SSIM')
    # plt.xlim([1,imgNum])

    plt.subplot(324)
    plt.plot(range(1, imgNum + 1), cc1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
    plt.title('CC')
    # plt.xlim([1,imgNum])

    plt.subplot(325)
    plt.plot(range(1, imgNum + 1), sf1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
    plt.title('SF')
    # plt.xlim([1,imgNum])

    plt.subplot(326)
    plt.plot(range(1, imgNum + 1), vif1, 'o-')
    plt.legend(range(epochbegin, epochbegin + imgNum - 1))
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

plt.savefig("." + imgDir + ".png")


# print(label)
# print(cc)
# print(sf)
# print(ssim)
# print(sd)
# print(en)
# print(vif)
