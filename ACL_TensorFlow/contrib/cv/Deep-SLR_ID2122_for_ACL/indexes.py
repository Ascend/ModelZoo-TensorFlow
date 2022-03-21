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

import numpy as np
import matplotlib.pyplot as plt
import auxiliaryFunctions as sf

nImg = 1
dispind = 0

# You can either load 'atb' and 'mask' from your saved binary files or use the pre-processing again. The latter is
# recommended (for we need 'org', which is not actually saved before).
test_data = sf.getTestingData(nImg=nImg)
org, _, atb, mask, std = test_data['org'], test_data['orgk'], test_data['atb'], test_data['mask'], test_data['minv']
# atb = np.fromfile('atb.bin', dtype=np.float32)
# mask = np.fromfile('mask.bin', dtype=np.complex64)

# Load the reconstruction result produced by model with right data type, and reshape it according to the output node
# shape.
rec = np.fromfile('deep-slr-model-100_output_0.bin', dtype=np.float32).reshape(1, 24, 256, 232, 1)

org = sf.create_sos(org)
atb = sf.create_sos(sf.r2c(atb) * std)
recon = sf.create_sos(sf.r2c(rec) * std)
error = np.abs(org - recon)
ssimrec = np.zeros((nImg,), dtype=np.float64)
psnrRec = np.zeros((nImg,), dtype=np.float64)
psnrAtb = np.zeros((nImg,), dtype=np.float64)
for i in range(nImg):
    ssimrec[i] = sf.ssimch(org[i], recon[i])
    psnrAtb[i] = sf.psnr(org[i], atb[i])
    psnrRec[i] = sf.psnr(org[i], recon[i])

print('  {0:.3f} {1:.3f} {2:.3f}  '.format(psnrAtb.mean(), psnrRec.mean(), ssimrec.mean()))

# Save the indexes to file
f = open('test_res.txt', 'w')
f.write('{0:.3f} {1:.3f} {2:.3f}'.format(psnrAtb.mean(), psnrRec.mean(), ssimrec.mean()))
f.close()

print('********************************')
recon = recon / recon.max()
error = error / error.max()
atb = atb / atb.max()
org = org / org.max()

# %% Display the output images
plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 0.8))
plt.clf()
plt.subplot(141)
st = 50
end = 220
plot(np.abs(org[dispind, st:end, :]))
plt.axis('off')
plt.title('Original')
plt.subplot(142)
plot(np.abs(atb[dispind, st:end, :]))
plt.title('Input \n PSNR=' + str(psnrAtb[dispind].round(2)) + ' dB')
plt.axis('off')
plt.subplot(143)
plot(np.abs(recon[dispind, st:end, :]))
plt.title('H-DSLR, PSNR=' + str(psnrRec[dispind].round(2)) + ' dB')
plt.axis('off')
plt.subplot(144)
plot(error[dispind, st:end, :])
plt.title('Error Image')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
plt.show()  # Only available for GUI

# Save the figure to file
plt.savefig('test_res.png')
