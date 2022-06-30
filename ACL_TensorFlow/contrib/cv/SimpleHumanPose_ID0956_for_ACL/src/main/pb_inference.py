from npu_bridge.npu_init import *
import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import math

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from model import Model

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms_my import oks_nms

output_dir = "cache/result"

msame_out_path = 'cache/dataset/outs'


def test_net(tester, dets, det_range, gpu_id):
    dump_results = []

    start_time = time.time()

    img_start = det_range[0]
    img_id = 0
    img_id2 = 0
    count = 0  # 计数
    total = 360
    pbar = tqdm(total=det_range[1] - img_start - 1, position=gpu_id)  # 进度条
    pbar.set_description("GPU %s" % str(gpu_id))
    while img_start < det_range[1]:
        img_end = img_start + 1
        im_info = dets[img_start]
        while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
            img_end += 1

        # all human detection results of a certain image
        cropped_data = dets[img_start:img_end]  # 一个确定的图片中的所有human检测结果

        pbar.update(img_end - img_start)
        img_start = img_end

        kps_result = np.zeros((len(cropped_data), cfg.num_kps, 3))
        area_save = np.zeros(len(cropped_data))

        # cluster human detection results with test_batch_size
        # test_batch_size 改成 1
        for batch_id in range(0, len(cropped_data), 1):
            start_id = batch_id
            end_id = min(len(cropped_data), batch_id + 1)

            imgs = []
            crop_infos = []
            for i in range(start_id, end_id):
                img, crop_info = generate_batch(cropped_data[i], stage='test')
                imgs.append(img)
                crop_infos.append(crop_info)

            imgs = np.array(imgs)
            crop_infos = np.array(crop_infos)

            # forward
            if 0 <= count < total:
                heatmap = np.loadtxt(osp.join(msame_out_path, "{}_output_0.txt".format(count)))
                heatmap = heatmap.reshape((1, 64, 48, 17))
                count += 1
                # print("count:", count)

            if cfg.flip_test:
                flip_imgs = imgs[:, :, ::-1, :]
                flip_heatmap = tester.predict_one([flip_imgs])[0]

                flip_heatmap = flip_heatmap[:, :, ::-1, :]
                for (q, w) in cfg.kps_symmetry:
                    flip_heatmap_w, flip_heatmap_q = flip_heatmap[:, :, :, w].copy(), flip_heatmap[:, :, :, q].copy()
                    flip_heatmap[:, :, :, q], flip_heatmap[:, :, :, w] = flip_heatmap_w, flip_heatmap_q
                flip_heatmap[:, :, 1:, :] = flip_heatmap.copy()[:, :, 0:-1, :]
                heatmap += flip_heatmap
                heatmap /= 2

            # for each human detection from clustered batch
            for image_id in range(start_id, end_id):

                for j in range(cfg.num_kps):
                    hm_j = heatmap[image_id - start_id, :, :, j]
                    idx = hm_j.argmax()
                    y, x = np.unravel_index(idx, hm_j.shape)

                    px = int(math.floor(x + 0.5))
                    py = int(math.floor(y + 0.5))
                    if 1 < px < cfg.output_shape[1] - 1 and 1 < py < cfg.output_shape[0] - 1:
                        diff = np.array([hm_j[py][px + 1] - hm_j[py][px - 1],
                                         hm_j[py + 1][px] - hm_j[py - 1][px]])
                        diff = np.sign(diff)
                        x += diff[0] * .25
                        y += diff[1] * .25
                    kps_result[image_id, j, :2] = (
                        x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
                    kps_result[image_id, j, 2] = hm_j.max() / 255

                # map back to original images
                for j in range(cfg.num_kps):
                    kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (
                            crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + \
                                                 crop_infos[image_id - start_id][0]
                    kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (
                            crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + \
                                                 crop_infos[image_id - start_id][1]

                area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (
                        crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])

        score_result = np.copy(kps_result[:, :, 2])
        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1, cfg.num_kps * 3)

        # rescoring and oks nms
        if cfg.dataset == 'COCO':
            rescored_score = np.zeros((len(score_result)))
            for i in range(len(score_result)):
                score_mask = score_result[i] > cfg.score_thr
                if np.sum(score_mask) > 0:
                    rescored_score[i] = np.mean(score_result[i][score_mask]) * cropped_data[i]['score']
            score_result = rescored_score
            keep = oks_nms(kps_result, score_result, area_save, cfg.oks_nms_thr)
            if len(keep) > 0:
                kps_result = kps_result[keep, :]
                score_result = score_result[keep]
                area_save = area_save[keep]
        elif cfg.dataset == 'PoseTrack':
            keep = oks_nms(kps_result, np.mean(score_result, axis=1), area_save, cfg.oks_nms_thr)
            if len(keep) > 0:
                kps_result = kps_result[keep, :]
                score_result = score_result[keep, :]
                area_save = area_save[keep]

        # save result
        for i in range(len(kps_result)):
            if cfg.dataset == 'COCO':
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(score_result[i], 4)),
                              keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'PoseTrack':
                result = dict(image_id=im_info['image_id'], category_id=1, track_id=0,
                              scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'MPII':
                result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())

            dump_results.append(result)

    return dump_results


def test(test_model):
    # annotation load
    d = Dataset()
    annot = d.load_annot(cfg.testset)  # ground truth
    gt_img_id = d.load_imgid(annot)  # 得到图像id

    # human bbox load
    if cfg.useGTbbox and cfg.testset in ['train', 'val']:
        if cfg.testset == 'train':
            dets = d.load_train_data(score=True)
        else:
            dets = d.load_val_data_with_annot()
        dets.sort(key=lambda x: (x['image_id']))
    else:
        with open(cfg.human_det_path, 'r') as f:
            dets = json.load(f)
        dets = [i for i in dets if i['image_id'] in gt_img_id]
        dets = [i for i in dets if i['category_id'] == 1]
        dets = [i for i in dets if i['score'] > 0]
        dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)

        img_id = []
        for i in dets:
            img_id.append(i['image_id'])
        imgname = d.imgid_to_imgname(annot, img_id, cfg.testset)  # 得到图像名
        for i in range(len(dets)):
            dets[i]['imgpath'] = imgname[i]  # 把图像名作为一项加入dets

    # job assign (multi-gpu)
    gpu_ids = "0"
    from tfflat.mp_utils_my import MultiProc
    img_start = 0
    ranges = [0]

    img_num = len(np.unique([i['image_id'] for i in dets]))

    images_per_gpu = int(img_num / len(gpu_ids.split(','))) + 1

    for run_img in range(img_num):
        img_end = img_start + 1
        while img_end < len(dets) and dets[img_end]['image_id'] == dets[img_start]['image_id']:
            img_end += 1
        if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num:
            ranges.append(img_end)
        img_start = img_end
    print("ranges:", ranges)

    global func

    def func(gpu_id):  # 修改
        # cfg.set_args(gpu_ids.split(',')[gpu_id])
        cfg.set_args()
        tester = Tester(Model(), cfg)
        tester.load_weights(test_model)
        range = [ranges[gpu_id], ranges[gpu_id + 1]]
        return test_net(tester, dets, range, gpu_id)

    MultiGPUFunc = MultiProc(len(gpu_ids.split(',')), func)
    result = MultiGPUFunc.work()

    # evaluation
    d.evaluation(result, annot, cfg.result_dir, cfg.testset)


if __name__ == '__main__':
    (npu_sess, npu_shutdown) = init_resource()
    test_epoch = 140
    test(test_epoch)
    shutdown_resource(npu_sess, npu_shutdown)
    close_session(npu_sess)
