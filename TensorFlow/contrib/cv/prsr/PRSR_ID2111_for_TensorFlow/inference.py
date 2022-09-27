from net import *
from data import *
from ops import *
from utils import *
import tensorflow as tf
import numpy as np

batch_size = 1

dataset = DataSet("./train.txt", 30, batch_size)
net = Net(dataset.hr_images, dataset.lr_images, 'prsr')

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()

# Create a session for running operations in the Graph.
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(init_op)
saver.restore(sess, './output/model.ckpt-280000')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

c_logits = net.conditioning_logits
p_logits = net.prior_logits
lr_imgs = dataset.lr_images
hr_imgs = dataset.hr_images
np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
gen_hr_imgs = np.zeros((batch_size, 32, 32, 3), dtype = np.float32)
np_c_logits = sess.run(c_logits, feed_dict = {lr_imgs: np_lr_imgs, net.train: False})

mu = 1.0
for i in range(32):
	for j in range(32):
		for c in range(3):
			np_p_logits = sess.run(p_logits, feed_dict = {hr_imgs: gen_hr_imgs})
			new_pixel = logits_2_pixel_value(
					np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256],
					mu = mu)
			gen_hr_imgs[:, i, j, c] = new_pixel
save_samples(gen_hr_imgs, './generate_imgs' + '.jpg')
save_samples(np_hr_imgs, './hr_imgs' + '.jpg')

import cv2
import numpy as np
import math

def psnr1(img1, img2):
	mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
	if mse < 1.0e-10:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img1 = cv2.imread("./generate_imgs.jpg")
img2 = cv2.imread("./hr_imgs.jpg")
print("PSNR is ", psnr1(img1, img2))