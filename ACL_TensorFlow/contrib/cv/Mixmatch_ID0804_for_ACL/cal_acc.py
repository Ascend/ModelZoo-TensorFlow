import numpy as np
import os
import pandas as pd

def read_excel(path, fix=1):
	l = []
	for file in os.listdir(path):
		# print(file)
		if file.endswith(".txt") :
			tmp = list(pd.read_csv(path + '//' + file))[0].split(' ')[:fix]
			tmp = np.asarray(tmp, dtype=float)
			l.append(tmp)
	return l

softmax = read_excel(r'/home/TestUser03/code/mixmatch/mix_model/out/20221010_16_50_33_510786', -1)
# print(softmax)
label = read_excel(r'/home/TestUser03/code/mixmatch/output_label_01')
# print(label)

crt = 0
for x,y in zip(softmax, label):
	pre_y = x.argmax()
	real_y = int(y[0])
	if pre_y == real_y:
		crt += 1

print("acc:", crt/len(label))