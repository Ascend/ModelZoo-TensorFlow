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

#!/usr/bin/env python
# coding: utf-8

# In[42]:


path=os.getcwd()
print(path) 


# In[175]:


import tensorflow as tf
import numpy as np
import pickle as pkl

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.float32) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)


# In[96]:


mnist_test[]


# In[112]:


mnist_test[:32].tofile("./out0/demo2.bin")


# In[110]:


mnist_test.shape


# In[109]:


mnist_test.dtype


# In[100]:


len(mnist_test)//32


# In[107]:


data= np.zeros((32,28,28,3))
data.astype("float32").tofile("./out0/demo.bin")


# In[108]:


data= np.zeros((1,28,28,3))
data.astype("float32").tofile("./out0/demo1.bin")


# In[122]:


output_path = "./out1/%d.bin"
j=0
for i in np.arange(311):
    mnist_test[j:j+32].astype(np.float32).tofile(output_path%i)
    j=j+32


# In[121]:


mnist_test[j:j+32].shape


# In[153]:


file_name = "./infer_result/%d_output_0.txt"
infer_label = []
for i in np.arange(311):
    t_file_name = file_name%i
    with open(t_file_name, "r") as f:  # 打开文件
        data=f.read()
        str=data.split()
        for char in str:
            infer_label.append(int(char))


# In[192]:


infer_label_np = np.asarray(infer_label)


# In[193]:


len(infer_label_np)


# In[194]:


infer_label_np[0:32]


# In[197]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

test_label=mnist.test.labels[0:9952]


# In[203]:


test_label[0:32]


# In[202]:


(infer_label_np == test_label).sum()/len(infer_label_np)


# In[204]:


np.save("./labels.npy", test_label)


# In[ ]:




