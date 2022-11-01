# Copyright 2020 Huawei Technologies Co., Ltd
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

from absl import flags
import sys
import om_bm_predict
import om_bm_predict_rnet
import om_bm_predict_onet
import cv2
import os
import numpy as np


# In[ ]:
FLAGS = flags.FLAGS
FLAGS(sys.argv)

out_path = "output/"
path = "picture/"
input_mode = "1"
if input_mode=='1':
    #选用图片
    
    for item in os.listdir(path):
        img_path=os.path.join(path,item)
        img=cv2.imread(img_path)
        os.system("python3 om_bm_predict.py {} {} {} {} {}".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]))
        os.system("python3 om_bm_predict_rnet.py {} {} {} {} {}".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[6]))
        boxes_c, landmarks = om_bm_predict_onet.do_bm_predict("{}".format(sys.argv[1]),"{}".format(sys.argv[2]),"{}".format(sys.argv[3]),"{}".format(sys.argv[4]),"{}".format(sys.argv[7]))
        os.system("rm -rf ge_*txt")
        print(boxes_c)
        for i in range(boxes_c.shape[0]):
            bbox=boxes_c[i,:4]
            score=boxes_c[i,4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            #画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            #判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score), 
                       (corpbbox[0], corpbbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        #画关键点
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])//2):
                cv2.circle(img, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))   
        cv2.imshow('im',img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:        
            cv2.imwrite(out_path + item,img)
    cv2.destroyAllWindows()


