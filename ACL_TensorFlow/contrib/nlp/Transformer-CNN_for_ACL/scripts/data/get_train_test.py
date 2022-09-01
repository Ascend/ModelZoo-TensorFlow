import csv
import numpy as np
import random


data = []
TRUE_LABEL = '/home/liangzixuan/Transformer-CNN/data/solubility_V1.csv'
first_row = True
for line in csv.reader(open(TRUE_LABEL, "r")):
    if first_row:
        first_row = False
        tmp = line
        continue
    data.append(line)

test = []
test_rate = 0.1
data_num = len(data)
for i in range(int(test_rate * data_num)):
    test.append(data.pop(random.randint(0, len(data))))



with open("test.csv", mode="w") as f:
    # 基于打开的文件，创建 csv.writer 实例
    writer = csv.writer(f)
    # 写入 header。
    # writerow() 一次只能写入一行。
    writer.writerow(tmp)
    # 写入数据。
    # writerows() 一次写入多行。
    writer.writerows(test)

with open("train.csv", mode="w") as f:
    # 基于打开的文件，创建 csv.writer 实例
    writer = csv.writer(f)
    # 写入 header。
    # writerow() 一次只能写入一行。
    writer.writerow(tmp)
    # 写入数据。
    # writerows() 一次写入多行。
    writer.writerows(data)

print("Relax!")