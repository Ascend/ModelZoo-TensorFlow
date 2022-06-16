import numpy as np
import time
import cv2

combine = []
img1 = cv2.imread('E:\Desktop\ShapeNetRendering/1111/test/rendering/00.png',cv2.IMREAD_COLOR)
img1 = cv2.resize(img1,(127,127))

img2 = cv2.imread('E:\Desktop\ShapeNetRendering/1111/test/rendering/01.png',cv2.IMREAD_COLOR)
img2 = cv2.resize(img2,(127,127))

img3 = cv2.imread('E:\Desktop\ShapeNetRendering/1111/test/rendering/02.png',cv2.IMREAD_COLOR)
img3 = cv2.resize(img3,(127,127))

combine = [img1,img2,img3]
combine = np.array(combine)
combine = combine.astype(np.float32)

combine.reshape(1,3,127,127,3)
combine = np.array([combine])
print(combine.shape)

combine.tofile('E:\Desktop\ShapeNetRendering/1111/test/rendering/combinefloat32.bin')