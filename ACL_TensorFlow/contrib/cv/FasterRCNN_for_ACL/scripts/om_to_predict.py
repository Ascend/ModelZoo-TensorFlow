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
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import sys,os,math
import numpy as np
import six



FLAGS = flags.FLAGS
FLAGS(sys.argv)


def do_bm_predict(path,datanum,batch,soureidpath,testdatapath):
        num = math.floor(int(datanum)/int(batch))
        model_out = {}
        predictions = []
        
        datanames = os.listdir(testdatapath)
        datanames.sort()

        for i in range(len(datanames)):
            try:
                if "images" in datanames[i]:
                    datanames[i] = datanames[i].split('_')                    
                    output1 = np.fromfile(os.path.join("{}".format(path), "{}_{}_image_info_output0.bin".format('davinci',datanames[i][0])),dtype=np.float32)
                    output2 = np.fromfile(os.path.join("{}".format(path), "{}_{}_image_info_output1.bin".format('davinci', datanames[i][0])),dtype=np.float32)
                    output3 = np.fromfile(os.path.join("{}".format(path), "{}_{}_image_info_output2.bin".format('davinci',datanames[i][0])),dtype=np.float32)
                    output4 = np.fromfile(os.path.join("{}".format(path), "{}_{}_image_info_output3.bin".format('davinci', datanames[i][0])),dtype=np.float32)
                    sourceid = np.fromfile(os.path.join("{}".format(soureidpath), "{}_source_ids.bin".format(datanames[i][0])),dtype=np.float32)
                    imginfo = np.fromfile(
                        os.path.join("{}".format(testdatapath), "{}_image_info.bin".format(datanames[i][0])),
                        dtype=np.float32)
                    output2 = output2.reshape(1, 100, 4)
                    output3 = output3.reshape(1, 100)
                    output4 = output4.reshape(1, 100)
                    imginfo = imginfo.reshape(1, 5)

                    predictions.append({
                        'num_detections': output1,  # 对应post_nms_num_valid_boxes，即valid_detections
                        'detection_boxes': output2,
                        'detection_classes': output3,
                        'detection_scores': output4,
                        'source_id':sourceid,
                        'image_info':imginfo,
                    })
                
            except e:
                print(e)
                print('iteraotr is null~~~~~')
                    
                break


        return predictions

if __name__ == '__main__':
    do_bm_predict()



