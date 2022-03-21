""" Rude Carnie: Age and Gender Deep Learning with Tensorflow found at
https://github.com/dpressel/rude-carnie
"""
# ==============================================================================
# MIT License
#
# Modifications copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multipreproc import FLAGS

from tensorflow.python.ops import array_ops
from datetime import datetime
import os
import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion
from pdb import set_trace as bp

VERSION_GTE_0_12_0 = LooseVersion(tf.__version__) >= LooseVersion('0.12.0')

# Name change in TF v 0.12.0
if VERSION_GTE_0_12_0:
	standardize_image = tf.image.per_image_standardization
else:
	standardize_image = tf.image.per_image_whitening

def data_files(data_dir, subset):
	"""Returns a python list of all (sharded) data subset files.
	Returns:
		python list of all (sharded) data set files.
	Raises:
		ValueError: if there are not data_files matching the subset.
	"""
	if subset not in ['train','validation','test']:
		print('Invalid subset!')
		exit(-1)

	tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
	data_files = tf.gfile.Glob(tf_record_pattern)
	print(data_files)
	if not data_files:
		print('No files found for data dir %s at %s' % (subset, data_dir))
		exit(-1)

	return data_files

def decode_jpeg(image_buffer, scope=None):
	"""Decode a JPEG string into one 3-D float image Tensor.
	Args:
	image_buffer: scalar string Tensor.
	scope: Optional scope for op_scope.
	Returns:
	3-D float Tensor with values ranging from [0, 1).
	"""

	with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
		# Decode the string as an RGB JPEG.
		# Note that the resulting image contains an unknown height and width
		# that is set dynamically by decode_jpeg. In other words, the height
		# and width of image is unknown at compile-time.
		image = tf.image.decode_jpeg(image_buffer, channels=3)

		# After this point, all image pixels reside in [0,1)
		# until the very end, when they're rescaled to (-1, 1).  The various
		# adjust_* ops all require this range for dtype float.
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)

		return image

def distort_image(image, height, width):
	# Image processing for training the network. Note the many random
	# distortions applied to the image.
	distorted_image = tf.random_crop(image, [height, width, 3])
	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	# Because these operations are not commutative, consider randomizing
	# the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

	return distorted_image

def _is_tensor(x):
	return isinstance(x, (tf.Tensor, tf.Variable))

def eval_image(image, height, width):
	return tf.image.resize_images(image, [height, width])

def data_normalization(image):

	image = standardize_image(image)

	return image

def image_preprocessing(image_buffer, image_size, train):
	"""Decode and preprocess one image for evaluation or training.
	Args:
	image_buffer: JPEG encoded string Tensor
	train: boolean
	thread_id: integer indicating preprocessing thread
	Returns:
	3-D float Tensor containing an appropriately scaled image
	Raises:
	ValueError: if user does not provide bounding box
	"""

	image = decode_jpeg(image_buffer)

	if train:
		# image = distort_image(image, image_size, image_size)
		image = eval_image(image, image_size, image_size)
	else:
		image = eval_image(image, image_size, image_size)

	image = data_normalization(image)
	return image

def multiparse_example_proto_train(example_serialized):
	# Dense features in Example proto.
	feature_map = {
		'image/encoded': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/ageclass/label': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/genderclass/label': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/height': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/width': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
	}

	features = tf.parse_single_example(example_serialized, feature_map)
	agelabel = tf.cast(features['image/ageclass/label'], dtype=tf.int32)
	genderlabel = tf.cast(features['image/genderclass/label'], dtype=tf.int32)
	image = image_preprocessing(features['image/encoded'], image_size= 227, train = True)
	return image, agelabel, genderlabel

def multiparse_example_proto_test(example_serialized):
	# Dense features in Example proto.
	feature_map = {
		'image/encoded': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/ageclass/label': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/genderclass/label': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
		'image/height': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
		'image/width': tf.FixedLenFeature([1], dtype=tf.int64,default_value=-1),
	}

	features = tf.parse_single_example(example_serialized, feature_map)
	agelabel = tf.cast(features['image/ageclass/label'], dtype=tf.int32)
	genderlabel = tf.cast(features['image/genderclass/label'], dtype=tf.int32)
	image = image_preprocessing(features['image/encoded'], image_size= 227, train = False)

	return image, agelabel, genderlabel

def multiinputs(data_dir,num_epochs, batch_size=128,train=True,):
	if train:
		dataset = tf.data.TFRecordDataset(data_dir)
		dataset = dataset.map(multiparse_example_proto_train)
		dataset = dataset.shuffle(500).repeat(num_epochs).batch(batch_size)
	else:
		dataset = tf.data.TFRecordDataset(data_dir)
		dataset = dataset.map(multiparse_example_proto_test)
		dataset = dataset.batch(batch_size)
	# iterator = dataset.make_one_shot_iterator()
	# images,agelabels, genderlabels, filenames = iterator.get_next()
	return dataset

