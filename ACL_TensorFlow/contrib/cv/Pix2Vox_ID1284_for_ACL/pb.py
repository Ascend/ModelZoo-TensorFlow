import tensorflow as tf
import os
#os.system('pip install PyMCubes')
import sys
sys.path.append('..')
import my_tools as tools
import numpy as np
import matplotlib.pyplot as plt

import argparse
import moxing as mox

#from create_tf_record import *
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, help='the path model saved')
parser.add_argument('--num_gpus', type=int, help='the number of gpu')
parser.add_argument("--data_url", type=str, default="./dataset")
obs_config = parser.parse_args()

data_dir = "/cache/dataset"
train_dir = "/cache/user-job-dir"
os.makedirs(data_dir)
mox.file.copy_parallel(obs_config.data_url, data_dir)

GPU='0'

batch_size = 6
img_res = 127
vox_res32 = 32
test_mv = 3
total_mv = 24

config={}                                 # python dictionary
config['batch_size'] = batch_size
config['total_mv'] = total_mv
config['test_mv'] = test_mv
#config['cat_names'] = ['02691156','02828884','04530566','03636649','03001627']
#config['cat_names'] = ['02691156','02828884','02933112','02958343','03001627','03211117',
#            '03636649','03691459','04090263','04256520','04379243','04401088','04530566']
config['cat_names'] = ['02691156']
for name in config['cat_names']:
    config['X_rgb_'+name] = '/cache/dataset/ShapeNetRendering/'+name+'/'

    config['Y_vox_'+name] = '/cache/dataset/ShapeNetVox32/'+name+'/'



def metric_iou(prediction, gt):
#    labels = tf.greater_equal(gt[gt], 0.5)
#    prediction = tf.cast(prediction,tf.int32)
    predictions = tf.greater_equal(prediction, 0.5)
    gt_=tf.greater_equal(gt, 0.5)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(predictions,gt_),tf.float32))
    union = tf.reduce_sum(tf.cast(tf.math.logical_or(predictions,gt_),tf.float32))
    iou = tf.cast(x = intersection,dtype=tf.float32)/ tf.cast(x = union,dtype=tf.float32)
    return iou


def freeze_graph(input_checkpoint,output_graph):
 
    output_node_names = "ref_net/ref_Dec/ref_out"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_graph_test(data, pb_path):

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
 
            input_pix_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
 
            Y_pred = tf.get_default_graph().get_tensor_by_name("ref_net/ref_Dec/ref_out:0")
            #Iou_ref = tf.get_default_graph().get_tensor_by_name("Evaluation_Metric/iou_refiner:0")
            #Iou_vae = tf.get_default_graph().get_tensor_by_name("Evaluation_Metric/iou_vae:0")

            data.shuffle_test_files(test_mv = 3, seed = 1)
            total_test_batch_num = data.total_test_batch_num  # int(len(self.X_rgb_train_files)/(self.batch_size*train_mv))
            print('total_test_batch_num:', total_test_batch_num)
            for i in range(total_test_batch_num):
                x_sample, gt_vox = data.load_X_Y_test_next_batch_sq(test_mv = 3)
                gt_vox = gt_vox.astype("float32")
                gt_vox = np.reshape(gt_vox, [batch_size, 32,32,32])

                y_pred = sess.run(Y_pred, feed_dict={input_pix_tensor: x_sample})
                iouref = metric_iou(y_pred, gt_vox)
                iouref = iouref.eval()
                print('iouref:', iouref)


#########################
if __name__ == '__main__':

    data = tools.Data(config)
    print("tools.data compleated")
#    ttest_demo(data,test_mv = 3)
    input_checkpoint='/cache/user-job-dir/code/train_mod/model.cptk'

    out_pb_path="frozen_model.pb"

    freeze_graph(input_checkpoint,out_pb_path)

    freeze_graph_test(data,out_pb_path)

    mox.file.copy_parallel(train_dir, obs_config.train_url)

    
