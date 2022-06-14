import cv2
import numpy as np
from keras.layers import Lambda
import tensorflow as tf

from skimage import io
import numpy as np
from numpy import array
from numpy.random import randint

from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   

def save2bin_1():
    a = cv2.imread("./t480.jpg")
    a = normalize(a)
    a.tofile("./48.bin")



def load_1():
    
    
    
    data = np.fromfile("./srgan_output_0.bin",np.float32).reshape(192,192,3)
    data = denormalize(data)
    cv2.imwrite("./opt.jpg",data)

save2bin_1()
#load_1()