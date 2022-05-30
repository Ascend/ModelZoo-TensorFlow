#!/usr/bin/env python
#title           :Utils.py
#description     :Have helper functions to process images and plot images
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf

from skimage import io
import numpy as np
from numpy import array
from numpy.random import randint
import cv2 as cv
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
    
# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images,dtype=object)
    return images_hr

# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale):

    images = []
    for img in  range(len(images_real)):
        images.append(cv.resize(images_real[img],(0,0),fx=0.25,fy=0.25,interpolation=cv.INTER_CUBIC))
    images_lr = array(images,dtype=object)
    return images_lr
    
def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   
 
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
    
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = io.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
                #if count%1000==0 :
                    #print(count)
    #print('finish load img')
    return files

def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files
    
def load_training_data(directory, ext, number_of_images = 10000, train_test_ratio = 0.8):

    number_of_train_images = int(number_of_images * train_test_ratio)
    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        #print("Number of image files are less then you specified")
        #print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    test_array = array(files,dtype=object)
    if len(test_array[0].shape) < 3:
        #print("Images are of not same shape")
        #print("Please provide same shape images")
        sys.exit()
    
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]
    
    x_train_hr = hr_images(x_train)
    for i in range(number_of_train_images):
        x_train_hr[i] = normalize(x_train_hr[i])
        #if i%1000==0:
            #print(i)
    #print("x_train_hr finish normalize")
    
    x_train_lr = lr_images(x_train, 4)
    for j in range(number_of_train_images):
        x_train_lr[j] = normalize(x_train_lr[j])    
        
    
    x_test_hr = hr_images(x_test)
    for k in range(number_of_images-number_of_train_images):
        x_test_hr[k] = normalize(x_test_hr[k])
        #if k %1000 == 0:
            #print(k)
    #print("x_test_hr finish normalize")

    
    x_test_lr = lr_images(x_test, 4)
    for w in range(number_of_images-number_of_train_images):
        x_test_lr[w] = normalize(x_test_lr[w])
        #if w %1000 == 0:
            #print(w)
    #print("x_test_lr finish normalize")

    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        #print("Number of image files are less then you specified")
        #print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)
    
    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)
    
    return x_test_lr, x_test_hr
    
def load_test_data(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        #print("Number of image files are less then you specified")
        #print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)
    
    return x_test_lr
    
# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]
    #print(examples)
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr,batch_size=16)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
    
    #plt.show()
    
# Plots and save generated images(in form LR, SR, HR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
    
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)
    
        #plt.show()

# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):
    
    examples = x_test_lr.shape[0]
    #image_batch_lr = denormalize(x_test_lr)
    image_batch_lr = x_test_lr
    #print(image_batch_lr.shape)
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    #print(generated_image.shape)
    for index in range(examples):
    
        #plt.figure(figsize=figsize)
    
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)
    
        #plt.show()





