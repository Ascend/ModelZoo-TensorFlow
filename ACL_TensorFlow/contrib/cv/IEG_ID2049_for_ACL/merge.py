import os
import numpy as np
import shutil


data_sets = []
sample_batch_input_bin_dir = "./imgout/"
for item in os.listdir(sample_batch_input_bin_dir):
    # 读取bin文件时，bin文件内的dtype类型须根据模型的输入类型确定，下面以float32为例
    original_input_data = np.fromfile(os.path.join(sample_batch_input_bin_dir, item), dtype=np.int32)
    # 将数据重新组织，具体根据模型输入中的shape值确定
    current_input_data = original_input_data.reshape(32, 32, 3)
    # 将当前的数据添加到列表中
    data_sets.append(current_input_data)
# 将每个batch的数据保存到一个输入bin文件中，从而得到一个包含多batch的输入bin文件
np.array(data_sets).tofile("newinput.bin")