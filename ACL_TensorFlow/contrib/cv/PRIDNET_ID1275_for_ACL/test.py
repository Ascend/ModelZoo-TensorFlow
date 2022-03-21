import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import importlib
import matplotlib.pyplot as plt
val_dir1='D:/Noisy raw-RGB data/ValidationGtBlocksRaw.mat'
val_dir2='D:/4000/ValidationCleanBlocksRaw.mat'
mat1 = loadmat(val_dir1)
mat2 = loadmat(val_dir2)

B1=mat1['ValidationGtBlocksRaw']
B2=mat2['results']
a,_=compare_ssim(B1,B2,full=True)     #SSIM
print('SSIM:')
print(a)
print('-----------------------------------')
print('PSNR:')
psnr =compare_psnr(B1, B2)               #PSNR
print(psnr)
