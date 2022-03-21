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

##Calculate Intersection Over Union Score for predicted layer
import numpy as np
import scipy.misc as misc

def GetIOU(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union

    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassIOU[i]))
       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight





