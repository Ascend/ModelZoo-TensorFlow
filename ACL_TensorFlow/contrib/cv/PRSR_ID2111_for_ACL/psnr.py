import cv2
import numpy as np
import math

def psnr1(img1, img2):
	mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
	if mse < 1.0e-10:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img1 = cv2.imread("./outimg/om_generate.jpg")
img2 = cv2.imread("./input_img/000001.jpg")

print("PSNR is ", psnr1(img1, img2))