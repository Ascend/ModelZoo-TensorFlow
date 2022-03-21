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

# Draw Train loss， please put the log in "train_log.txt" first中
import matplotlib.pyplot as plt

x = []
y = []
with open("./train_log.txt", "r", encoding="utf-8") as f:
    lline = f.readlines()
    index = 0
    for ll in lline:
        if ll.find("step - loss:") >= 0:
            x.append(index)
            index += 1
            value = ll.split("step - loss:")[-1]
            if value.find("- val_loss: ") > 0:
                value = value.split("- val_loss:")[0].strip()
            else:
                value = value.strip()
            y.append(float(value))
plt.title('train loss')
plt.plot(x, y, color='green', label='training loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
