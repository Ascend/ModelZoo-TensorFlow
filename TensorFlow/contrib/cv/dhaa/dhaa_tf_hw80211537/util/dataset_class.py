"""
Function: Class CASIA_Race
Format: Race/Race-ID-IdCard/Race_ID_AcqDevice_Session_PAI
    Race: 1:AF 2:CA 3:EA
    ID: 000-599
    AcqDevice: 1:rssdk 2:mp4  3:bag 4:MOV
    Session(environment): 1:indoor 2:outdoor 3:random
    PAI: 1:Real 2:Cloth 3:Pic(phtoto) 4:Screen
Example: EA/EA-012-198006250024/1_000_1_1_1(P1_P2_P3_P4_P5)
Info:
AF: Num = 2,094,730, RAM = 260G
CA: Num = 2,025,816, RAM = 269G
EA: Num = 1,884,592, RAM = 244G

Author: AJ
Date: 2019.9.27
"""

import os, copy, glob, random, time, sys
# from protocol_class import *
import util.utils as utils
import numpy as np

ori_key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
flip_key = [2, 1, 6, 5, 4, 3, 7, 8, 9, 10, 12, 11, 14, 13, 15, 16, 22, 21, 20, 19, 18, 17, 24, 23, 25]
ori_key = [x - 1 for x in ori_key]
flip_key = [x - 1 for x in flip_key]

def lms_flip(lms_infor_ori):
    lms_infor_flip = lms_infor_ori * 1.0
    for key in range(lms_infor_flip.shape[0]):
        lms_infor_flip[key,:] = lms_infor_ori[flip_key[key], :]
        lms_infor_flip[key][0] = 224 - lms_infor_flip[key][0]
    return lms_infor_flip

def get_ped_dataset(path, train_file, with_lms=False):
    dataset = []
    with open(train_file) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            image_path = os.path.join( path, contents[0])
            # labels = [int(x) for x in contents[1:]]
            # labels = np.array( labels, dtype=np.int )
            labels = [int(contents[1])]
            # load landmarks
            lms_infor = np.zeros((25, 2), )
            shape_file = image_path + '.SDMshape2'
            with open(shape_file) as f_a:
                id = 0
                for eachlineS in f_a:
                    contents = eachlineS.strip().split(' ')
                    if len(contents) == 3:
                        # lms_infor[int(contents[0])] = [float(contents[1]), float(contents[2])]
                        lms_infor[id, 0] = float(contents[1])
                        lms_infor[id, 1] = float(contents[2])
                        id += 1
            assert len(lms_infor) == 25
            assert id == 25
            lms_infor_flip = lms_flip(lms_infor)
            lms_infor_two = np.stack((lms_infor, lms_infor_flip), axis=0)
            dataset.append((image_path, labels, lms_infor_two))
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    thetas_flat = []
    lms_flat = []
    for i in range(len(dataset)):
        image_paths_flat += [dataset[i][0]]
        labels_flat += [dataset[i][1]]
        #     thetas_flat += [get_trans_theta(dataset[i][2])]
        # return image_paths_flat, labels_flat, thetas_flat
        lms_flat += [dataset[i][2]]
    return image_paths_flat, labels_flat, lms_flat
