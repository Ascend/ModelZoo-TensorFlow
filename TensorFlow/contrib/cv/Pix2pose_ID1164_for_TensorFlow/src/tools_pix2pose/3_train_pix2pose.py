# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
import os,sys
from npu_bridge.npu_init import *
# os.system("pip install --upgrade pip")
# print("-----------------------------------------")
# os.system("pip list")
# print("-----------------------------------------")
# os.system("pip install transforms3d")
# print("-----------------------------------------")
# os.system("pip install pypng")
# print("-----------------------------------------")
# os.system("pip install keras==2.2.4")
# print("-----------------------------------------")

import transforms3d as tf3d
from math import radians
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--obj_id", type=str, default="01")
parser.add_argument("--output_path", type=str, default="/home/ma-user/modelarts/inputs/data_url_0/")
parser.add_argument("--data_path", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")
parser.add_argument("--back_dir", type=str, default="./tless/train2017/train2017/") #add
parser.add_argument("--max_epoch_case", type=int, default=0)#add  epoch控制
parser.add_argument("--n_batch_per_epoch_case", type=int, default=0)#add   steps控制
config = parser.parse_args()

#参数设置
dataset = 'tless'
cfg_fn = './src/tools_pix2pose/cfg_tless_paper.json'
back_dir  = config.back_dir
gpu_id ='0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
obj_id = config.obj_id

#没啥用
modelarts_data_dir =config.data_path+dataset
modelarts_output_dir =config.output_path


ROOT_DIR = os.path.abspath("..")
ROOT_DIR = os.path.abspath("./src/")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./src/bop_toolkit_lib")

from bop_toolkit_lib import inout,dataset_params

from pix2pose_model import ae_model as ae
import matplotlib.pyplot as plt
import time
import random
import numpy as np

import tensorflow as tf
from keras import losses
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint,Callback
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.utils import GeneratorEnqueuer
from keras.layers import Layer

from pix2pose_util import data_io as dataio
from tools_pix2pose import bop_io

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)


def dummy_loss(y_true,y_pred):
    return y_pred

def get_disc_batch(X_src, X_tgt, generator_model, batch_counter,label_smoothing=False,label_flipping=0):    
    if batch_counter % 2 == 0:        
        X_disc,prob_dummy = generator_model.predict(X_src,batch_size=50)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        if label_smoothing:
            y_disc = np.random.uniform(low=0.0, high=0.1, size=y_disc.shape[0])            
        else:
            y_disc = 0
            
        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:] = 1
    else:
        X_disc = X_tgt
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc = np.random.uniform(low=0.9, high=1.0, size=y_disc.shape[0])                
        else:
            y_disc = 1                
        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:] = 0

    return X_disc, y_disc



loss_weights = [100,1]
train_gen_first = False
load_recent_weight = True




 #"cfg/cfg_bop2019.json"

cfg = inout.load_json(cfg_fn)
bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True)


print(bop_dir)
print("--------------------")
print(source_dir)
im_width,im_height =cam_param_global['im_size']
weight_prefix = "pix2pose" 
obj_id = int(config.obj_id) #identical to the number for the ply file.
#weight_dir = cfg['dataset_dir']+"/tless/pix2pose_weights/{:02d}".format(obj_id)  #add
weight_dir = "./pix2pose_weights/{:02d}".format(obj_id)  #add---------避免改动数据集的大小
if not(os.path.exists(weight_dir)):
        os.makedirs(weight_dir)

data_dir = bop_dir+"/train_xyz/{:02d}".format(obj_id)

batch_size=50
datagenerator = dataio.data_generator(data_dir,back_dir,batch_size=batch_size,res_x=im_width,res_y=im_height)

m_info = model_info['{}'.format(obj_id)]
keys = m_info.keys()
sym_pool=[]
sym_cont = False
sym_pool.append(np.eye(3))
if('symmetries_discrete' in keys):
    print(obj_id,"is symmetric_discrete")
    print("During the training, discrete transform will be properly handled by transformer loss")
    sym_poses = m_info['symmetries_discrete']
    print("List of the symmetric pose(s)")
    for sym_pose in sym_poses:
        sym_pose = np.array(sym_pose).reshape(4,4)
        print(sym_pose[:3,:3])
        sym_pool.append(sym_pose[:3,:3])
if('symmetries_continuous' in keys):
    sym_cont=True

optimizer_dcgan =Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
optimizer_disc = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
backbone='paper'
if('backbone' in cfg.keys()):
    if(cfg['backbone']=="resnet50"):
            backbone='resnet50'
if(backbone=='resnet50'):
    generator_train = ae.aemodel_unet_resnet50(p=1.0)
else:
    generator_train = ae.aemodel_unet_prob(p=1.0)
    

discriminator = ae.DCGAN_discriminator()
imsize=128
dcgan_input = Input(shape=(imsize, imsize, 3))
dcgan_target = Input(shape=(imsize, imsize, 3))
prob_gt = Input(shape=(imsize, imsize, 1))
gen_img,prob = generator_train(dcgan_input)
recont_l = ae.transformer_loss(sym_pool)([gen_img,dcgan_target,prob,prob_gt])
discriminator.trainable = False
disc_out = discriminator(gen_img)
dcgan = Model(inputs=[dcgan_input,dcgan_target,prob_gt],outputs=[recont_l,disc_out])

epoch=0
recent_epoch=-1

if load_recent_weight:
    weight_save_gen=""
    weight_save_disc=""
    for fn_temp in sorted(os.listdir(weight_dir)):
        if(fn_temp.startswith(weight_prefix+".")):
                    temp_split  = fn_temp.split(".")
                    epoch_split = temp_split[1].split("-") #"01_real_1.0-0.1752.hdf5"
                    epoch_split2= epoch_split[0].split("_") #01_real_1.0
                    epoch_temp = int(epoch_split2[0])
                    network_part = epoch_split2[1]
                    if(epoch_temp>=recent_epoch):
                        recent_epoch = epoch_temp
                        if(network_part=="gen"):                        
                            weight_save_gen = fn_temp
                        elif(network_part=="disc"):
                            weight_save_disc = fn_temp

    if(weight_save_gen!=""):
        print("load recent weights from", weight_dir+"/"+weight_save_gen)
        generator_train.load_weights(os.path.join(weight_dir,weight_save_gen))
    
    if(weight_save_disc!=""):
        print("load recent weights from", weight_dir+"/"+weight_save_disc)
        discriminator.load_weights(os.path.join(weight_dir,weight_save_disc))
   

if(recent_epoch!=-1):
    epoch = recent_epoch
    train_gen_first=False
max_epoch=10
if(max_epoch==10): #lr-shcedule used in the bop challenge
    lr_schedule=[1E-3,1E-3,1E-3,1E-3,1E-3,
                1E-3,1E-3,1E-4,1E-4,1E-4,
                1E-5,1E-5,1E-5,1E-5,1E-6,
                1E-6,1E-6,1E-6,1E-6,1E-7]
elif(max_epoch==20): #lr-shcedule used in the paper
    lr_schedule=[1E-3,1E-3,1E-3,1E-3,1E-3,
                1E-3,1E-3,1E-3,1E-3,1E-4,
                1E-4,1E-4,1E-4,1E-4,1E-4,
                1E-4,1E-4,1E-4,1E-4,1E-5]

dcgan.compile(loss=[dummy_loss, 'binary_crossentropy'],
                loss_weights=loss_weights ,optimizer=optimizer_dcgan)
dcgan.summary()

discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer_disc)
discriminator.summary()

N_data=datagenerator.n_data
batch_size= 50
batch_counter=0
n_batch_per_epoch= min(N_data/batch_size*10,3000) #check point: every 10 epoch  #2096
step_lr_drop=5
perf_list=[]
fps_list=[]
disc_losses=[]
recont_losses=[]
gen_losses=[]
pre_loss=9999
lr_current=lr_schedule[epoch]

real_ratio=1.0
feed_iter= datagenerator.generator()
K.set_value(discriminator.optimizer.lr, lr_current)
K.set_value(dcgan.optimizer.lr, lr_current)
fed = GeneratorEnqueuer(feed_iter,use_multiprocessing=True, wait_time=5)
fed.start(workers=6,max_queue_size=200)
iter_ = fed.get()

zero_target = np.zeros((batch_size))
for X_src,X_tgt,disc_tgt,prob_gt in iter_:
    start_time = time.time()
    discriminator.trainable = True
    X_disc, y_disc = get_disc_batch(X_src,X_tgt,generator_train,0,
                                    label_smoothing=True,label_flipping=0.2)
    disc_loss = discriminator.train_on_batch(X_disc, y_disc)

    X_disc, y_disc = get_disc_batch(X_src,X_tgt,generator_train,1,
                                    label_smoothing=True,label_flipping=0.2)
    disc_loss2 = discriminator.train_on_batch(X_disc, y_disc)
    disc_loss  = (disc_loss + disc_loss2)/2

    discriminator.trainable = False

    dcgan_loss = dcgan.train_on_batch([X_src,X_tgt,prob_gt],[zero_target,disc_tgt])

    disc_losses.append(disc_loss)
    recont_losses.append(dcgan_loss[1])
    gen_losses.append(dcgan_loss[2])

    mean_loss = np.mean(np.array(recont_losses))
    if batch_counter > 1: #去掉第一次数据
        perf = time.time() - start_time
        perf_list.append(perf)
        perf_ = np.mean(perf_list)
        fps = batch_size / perf
        fps_list.append(fps)
        fps_ = np.mean(fps_list)
        print("Epoch{:02d}-Iter{:03d}/{:03d}: Mean- {:.5f} , perf- {:.4f} , fps- {:.4f} ,  Disc- {:.4f} ,  Recon-[{:.4f}], Gen-[{:.4f}]],lr={:.6f}".format(epoch,batch_counter,int(n_batch_per_epoch),mean_loss,perf_,fps_,disc_loss,dcgan_loss[1],dcgan_loss[2],lr_current))
    if(batch_counter>(n_batch_per_epoch-config.n_batch_per_epoch_case)):  #add 控制步数，缩短性能看护时间
        mean_loss = np.mean(np.array(recont_losses))
        disc_losses=[]
        recont_losses=[]
        gen_losses=[]
        batch_counter=0
        epoch+=1
        print('disc_loss:',disc_loss)
        print('dcgan_loss:',dcgan_loss)
        if( mean_loss< pre_loss):
            print("loss improved from {:.4f} to {:.4f} saved weights".format(pre_loss,mean_loss))
            print(weight_dir+"/"+weight_prefix+".{:02d}-{:.4f}.hdf5".format(epoch,mean_loss))
            pre_loss=mean_loss
        else:
            print("loss was not improved")
            print(weight_dir+"/"+weight_prefix+".{:02d}-{:.4f}.hdf5".format(epoch,mean_loss))

        weight_save_gen = weight_dir+"/" + weight_prefix+".{:02d}_gen_{:.1f}-{:.4f}.hdf5".format(epoch,real_ratio,mean_loss)
        weight_save_disc = weight_dir+"/" + weight_prefix+".{:02d}_disc_{:.1f}-{:.4f}.hdf5".format(epoch,real_ratio,mean_loss)
        generator_train.save_weights(weight_save_gen)
        discriminator.save_weights(weight_save_disc)
        
        gen_images,probs = generator_train.predict(X_src,batch_size=50)

        imgfn = weight_dir+"/val_img/"+weight_prefix+"_{:02d}.png".format(epoch)
        if not(os.path.exists(weight_dir+"/val_img/")):
            os.makedirs(weight_dir+"/val_img/")
        
        f,ax=plt.subplots(10,3,figsize=(10,20))
        for i in range(10):
            ax[i,0].imshow( (X_src[i]+1)/2)
            ax[i,1].imshow( (X_tgt[i]+1)/2)
            ax[i,2].imshow( (gen_images[i]+1)/2)
        plt.savefig(imgfn)
        plt.close()
        
        lr_current=lr_schedule[epoch]
        K.set_value(discriminator.optimizer.lr, lr_current)
        K.set_value(dcgan.optimizer.lr, lr_current)        

    batch_counter+=1
    if(epoch>(max_epoch-config.max_epoch_case)):  #add 控制变量
        print("Train finished")
        if(backbone=='paper'):
            generator_train.save_weights(os.path.join(weight_dir,"inference.hdf5"))        
        else:
            generator_train.save(os.path.join(weight_dir,"inference_resnet_model.hdf5"))        
        break

sess.close()