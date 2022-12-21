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
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_text_len(file):
    """
    文本长度可视化
    :param file:
    :return:
    """
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    # 获取所有文本的token 和 char 特征长度
    all_length = [(len(i.strip().split("\t")[1].split(" ")), len(i.strip().split("\t")[2].split(" "))) for i in lines]
    all_token_length = [i[0] for i in all_length]
    all_char_length = [i[1] for i in all_length]
    # print(all_token_length[:2])
    # 可视化语料序列长度, 可见大部分文本的长度都在1000以下
    token_prop = np.mean(np.array(all_token_length) < 1000)
    print("文本token长度在1000以下的比例: {}".format(token_prop))
    plt.hist(all_token_length, bins=500)
    plt.show()

    char_prop = np.mean(np.array(all_char_length) < 2000)
    print("文本char长度在2000以下的比例: {}".format(char_prop))
    plt.hist(all_token_length, bins=500)
    plt.show()


def main():
    file = "../data/train.txt"
    plot_text_len(file)


if __name__ == "__main__":
    main()