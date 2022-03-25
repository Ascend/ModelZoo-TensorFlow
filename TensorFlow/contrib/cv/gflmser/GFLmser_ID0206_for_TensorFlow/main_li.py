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

import tensorflow as tf
import sys
import os
import glob
from os.path import join
import datetime
import shutil
import math
import numpy as np
import cv2 as cv
#import moxing as mox
import time
import GFLmser
import data
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
# os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
# os.environ['ASCEND_GLOBAL_EVENT_ENABLE'] = '1'
# os.environ['GE_USE_STATIC_MEMORY'] = "1"

# BASE_DIR = sys.path[0]
# PROJECT_DIR = os.path.dirname(BASE_DIR)
# sys.path += [PROJECT_DIR, BASE_DIR]
# print(PROJECT_DIR, '\n', BASE_DIR)


tf.flags.DEFINE_integer('batch_size', 140,'')
tf.flags.DEFINE_integer('print_interval', 300,'')
tf.flags.DEFINE_float('lr_up', 0.0001, '')
tf.flags.DEFINE_float('lr_down', 0.0001, '')
tf.flags.DEFINE_float('lr_dis', 0.0001, '')
tf.flags.DEFINE_string('ckpt_dir', 'new_ckpt', '')
tf.flags.DEFINE_string('data_url', '/home/jnn/nfs/mnist', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/home/jnn/temp/delete', 'saved model directory.')
tf.flags.DEFINE_string('train_dir', '/home/linshihan/vc_workspace/GMLmser/dataset/sr_train', '')
tf.flags.DEFINE_string('test_dir', '/home/linshihan/vc_workspace/GMLmser/dataset/sr_test', '')
tf.flags.DEFINE_float('beta', 0.05, '')
tf.flags.DEFINE_integer('gpu', 1, '')
tf.flags.DEFINE_integer('is_resume', 0, '')
tf.flags.DEFINE_integer('epochs', 300, '') #add
tf.flags.DEFINE_integer('batch_num_less', 0, '') #add
tf.flags.DEFINE_string('data_dir', './data', '')#add

args = tf.flags.FLAGS

TMP_CACHE_PATH = args.data_dir #add
#mox.file.copy_parallel(args.data_url, TMP_CACHE_PATH)
args.train_dir = join(TMP_CACHE_PATH, 'train')
args.test_dir = join(TMP_CACHE_PATH, 'test')

# args.ckpt_dir = join(BASE_DIR, args.ckpt_dir)
# if not os.path.exists(args.ckpt_dir):
#     os.mkdir(args.ckpt_dir)


def get_psnr(img1, img2, max_val):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        print(100)
    if max_val == 1:
        psnr = 10 * math.log10(1. / mse)
    elif max_val == 255:
        psnr = 20 * math.log10(255. / math.sqrt(mse))
    else:
        raise ValueError
    return psnr


def save_model(sess, saver, name='model', global_steps=0):
    saver.save(sess, os.path.join(args.ckpt_dir, name), global_step=global_steps)


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def delete_ckpt():
    if os.path.exists(args.ckpt_dir):
        shutil.rmtree(args.ckpt_dir)
        print('===> deleted previous checkpoint !')


def test(sess, model, test_set):
    sum_psnr = 0
    cnt = 0
    input_img, lr_img, hr_img = test_set.get_data()
    batch_num = math.floor(input_img.shape[0]/args.batch_size)
    batch_id = 0
    for step in range(1,int(batch_num)+1):
        t_input_img, t_lr_img, t_hr_img, batch_id = test_set.get_next(batch_id, args.batch_size, input_img, lr_img, hr_img)
        feed_dict = {model.input_data:t_input_img, model.lr_img:t_lr_img, model.hr_img: t_hr_img, model.training:False}
        fake_lr_out, fake_hr_out, batch_psnr = sess.run([model.fake_lr, model.test_out_hr, model.PSNR], feed_dict)
        # batch_psnr = get_psnr(fake_hr_out, t_hr_img, max_val=1.0)

        sum_psnr += batch_psnr
        cnt += 1 
    avg_psnr = sum_psnr / cnt
    return avg_psnr


def train():
    graph = tf.Graph()
    with graph.as_default():
        model = GFLmser.GFLmser(args)
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log",graph = graph)
        session_config = tf.ConfigProto()
        #os.mkdir("/cache/profiling")
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["graph_memory_max_size"].s = b'21474836480'
        custom_op.parameter_map["variable_memory_max_size"].s = b'11811160064'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        
        session = tf.Session(graph=graph, config = session_config)

        session.run(tf.global_variables_initializer())
    global_step = 0
    start_epoch = 0
    max_psnr = 0
    npy = 0
    print('===> Training')
    test_set = data.LmserData_test(args.test_dir)

    for epo in range(start_epoch, args.epochs):
        train_npy_path = glob.glob(join(args.train_dir, 'G_*.npy'))
        for npy, npy_path in enumerate(train_npy_path):  #11
            data_train = data.LmserData_train(npy_path, args.train_dir)
            d_data, input_img, lr_img, hr_img = data_train.get_data() #delete
            batch_num = math.floor(input_img.shape[0]/args.batch_size)
            batch_id = 0
            print(get_time(), npy, npy_path)
            print("int(batch_num)-------------",int(batch_num))  #142
            for step in range(1, int(batch_num)+1 - args.batch_num_less):   #原来：142
                print(get_time(), 'step', step)
                g_d_data, g_input_img, g_lr_img, g_hr_img, batch_id = data_train.get_next(batch_id, args.batch_size, d_data, input_img, lr_img, hr_img)
                feed_dict = {model.real_lr: g_d_data, model.input_data: g_input_img, model.lr_img: g_lr_img, model.hr_img:g_hr_img, model.training: True}
                start = time.time()
                _up, up_mse_loss = session.run([model.up_train_op, model.up_mse_loss], feed_dict)
                _dis, d_cost = session.run([model.dis_train_op, model.discrim_cost], feed_dict)
                perf = time.time() - start
                fps = args.batch_size / perf
                print("time: {:.4f} fps: {:.4f}".format(perf,fps))
                
                if step % 2 == 0:
                    _down, down_mse_loss = session.run([model.down_train_op, model.down_mse_loss], feed_dict)
                global_step += 1

                if global_step % args.print_interval == 1:  #300
                    psnr = test(session, model, test_set)
                    d_cost, g_cost, down_mse_loss, rs = session.run([ model.discrim_cost, model.generator_cost, model.down_mse_loss,merged], feed_dict)
                    with open(os.path.join(args.ckpt_dir, 'log.txt'), 'a', encoding='utf-8') as f:
                        s = str(get_time()) +" "+ "epo:{} npy:{} D_cost: {:.5f} G_cost:{:.5f} up_mse:{:.5f} down_mse:{:.5f} psnr:  {}".format(epo, npy, d_cost, g_cost, up_mse_loss, down_mse_loss, psnr)
                        f.write(s + '\n')
                    
                    writer.add_summary(rs, global_step)
                    fake_lr_img, fake_hr_img, g_hr_img = session.run([model.fake_lr, model.train_out_hr, model.test_out_hr],feed_dict)

                    save_img(fake_lr_img, "fake_lr_img_{}".format(global_step), 'fake_lr_img')
                    save_img(fake_hr_img, "fake_hr_img_{}".format(global_step), 'fake_hr_img')
                    save_img(g_hr_img, "g_hr_img_{}".format(global_step), 'g_hr_img')
                    save_img(g_input_img, "input_img_{}".format(global_step), 'input_img')
                    print(get_time(), "epo:{} npy:{} D_cost: {:.5f} G_cost:{:.5f} up_mse:{:.5f} down_mse:{:.5f} psnr:{}".format(epo, npy, d_cost, g_cost, up_mse_loss, down_mse_loss, psnr))

                    if max_psnr < psnr:
                        print(psnr, '------------- sota found -------------')
                        max_psnr = psnr
                        save_name = 'epo_{}_npy_{}_psnr_{}'.format(epo, npy, round(psnr, 4))
                        save_model(session, saver, save_name, global_steps=global_step)

    session.close()
    #mox.file.copy_parallel("/var/log/npu/", args.train_url)


def save_img(imgs, img_name, _type):
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    img_dir = join(args.ckpt_dir, _type)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for i in range(1):
        img = np.clip((imgs[i] +1 ) / 2. * 255., a_min=0., a_max=255.)
        # print('img.shape=', img.shape)
        cv.imwrite(join(img_dir, img_name + '.jpg'), img[:, :, ::-1])
    # print('saved imgs')


if __name__ == "__main__":
    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    train()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))