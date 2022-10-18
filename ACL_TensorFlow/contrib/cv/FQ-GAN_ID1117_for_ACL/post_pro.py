from email.mime import image
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os, random
import numpy as np
import argparse
from glob import glob



class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag: 
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img

def load_test_data(image_path, size=256):
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(size, size))

    img = np.expand_dims(img, axis=0)
    img = img/127.5 - 1

    return img

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')
def parse_args():
    desc = "post prosess of fake_img"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', type=str, default='/home/test_user07/FQ-GAN/20221016_18_13_56_414701', 
    help='path of msame output')
    return parser.parse_args()  
def process(args):
    print(args.input)
    files = glob('{}/*1.txt'.format(args.input))
    output_dir=os.path.join(args.input, "img_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(len(files))
    for sample_file  in files:
        a=np.loadtxt(sample_file)
        a=np.reshape(a, (1, 256, 256, 3))
        file_name=os.path.basename(sample_file)
        file_name=file_name.split('.')[0][: -2]
        image_path=os.path.join(args.input+"/img_output", file_name+".jpg")
        print(image_path)
        save_images(a, [1, 1], image_path)
    print("done")
def main():
    args = parse_args()
    process(args)
if __name__ == '__main__':
    main()