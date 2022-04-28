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
import tensorflow as tf
from config import args
from getData.voc_loader import VOCLoader

from tensorflow.python.ops.metrics_impl import mean_iou as streaming_mean_iou
from utils import decode_bboxes
from getData.boxer import PriorBoxGrid
from config import config as net_config
from detector import Detector
from tabulate import tabulate
import progressbar
import numpy as np
import logging
log = logging.getLogger()

def eval_category(gt, dets, cid):
    """Computes average precision for one category"""
    cgt = gt[cid]
    cdets = np.array(dets[cid])
    if (cdets.shape == (0, )):
        return None, None
    scores = cdets[:, 1]
    sorted_inds = np.argsort(-scores)
    image_ids = cdets[sorted_inds, 0].astype(int)
    BB = cdets[sorted_inds]

    npos = 0
    for img_gt in cgt.values():
        img_gt['ignored'] = np.array(img_gt['difficult'])
        img_gt['det'] = np.zeros(len(img_gt['difficult']), dtype=np.bool)
        npos += np.sum(~img_gt['ignored'])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        ovmax = -np.inf
        if image_ids[d] in cgt:
            R = cgt[image_ids[d]]
            bb = BB[d, 2:].astype(float)

            BBGT = R['bbox'].astype(float)

            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 0] + BBGT[:, 2], bb[0] + bb[2])
            iymax = np.minimum(BBGT[:, 1] + BBGT[:, 3], bb[1] + bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih

            # union
            uni = (bb[2] * bb[3] + BBGT[:, 2] * BBGT[:, 3] - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > args.voc_iou_thresh:
            if not R['ignored'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = True
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    N = float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = rec * N / np.maximum(rec * N + fp, np.finfo(np.float32).eps)
    return rec, prec

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_ap(gt, dets, loader):
    """computes average precision for all categories"""
    aps = {}
    for cid in range(1, loader.num_classes):
        cat_name = loader.ids_to_cats[cid]
        rec, prec = eval_category(gt, dets, cid)
        ap = voc_ap(rec, prec, loader.year == '07')
        aps[loader.ids_to_cats[cid]] = ap
    return aps

def make_detection_table(gt, dets, loader):
    """creates a table with AP per category and mean AP"""
    aps = compute_ap(gt, dets, loader)
    print("ap = ", aps)
    eval_cache = [aps]

    table = []
    for cid in range(1, loader.num_classes):
        cat_name = loader.ids_to_cats[cid]
        table.append((cat_name, ) + tuple(aps.get(cat_name, 'N/A') for aps in eval_cache))
    mean_ap = np.mean([a for a in list(aps.values()) if a >= 0])
    table.append(("AVERAGE", ) + tuple(np.mean(list(aps.values())) for aps in eval_cache))
    x = tabulate(table, headers=(["Category", "mAP (all)"]),
                 tablefmt='orgtbl', floatfmt=".3f")
    log.info("Eval results:\n%s", x)
    return table

def compute_mean_iou(detector):
    iou = detector.get_mean_iou()
    print(iou)
    log.info("\n Mean IoU is %f", iou)
    return iou

def main(argv=None):
    if args.dataset == 'voc07' or args.dataset == 'voc07+12':
        loader = VOCLoader('07', 'test')
    if args.dataset == 'voc12-val':
        loader = VOCLoader('12', 'val', segmentation=args.segment)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        detector = Detector(sess, loader, net_config, no_gt=args.no_seg_gt)

        filenames = loader.get_filenames()
        gt = {cid: {} for cid in range(1, loader.num_classes)}
        dets = {cid: [] for cid in range(1, loader.num_classes)}

        bar = progressbar.ProgressBar()# 显示进度条
        # print("filenames = ", filenames)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        for i in bar(range(len(filenames))):
            name = filenames[i]
            # print("name = ", name)
            img_id = i
            img = loader.load_image(name) # 获取图片
            # img = np.fromfile("./binFile/img/{0:05d}.bin".format(i), dtype=np.float32)
            # img.shape = 1, 300, 300, 3
            gt_bboxes, seg_gt, gt_cats, w, h, difficulty = loader.read_annotations(name) # 获取图片信息

            confidence = np.loadtxt("./binFile/test/2021118_18_51_25_234650/{0:05d}_output_0.txt".format(i))
            location = np.loadtxt("./binFile/test/2021118_18_51_25_234650/{0:05d}_output_1.txt".format(i))
            seg_logits = np.loadtxt("./binFile/test/2021118_18_51_25_234650/{0:05d}_output_2.txt".format(i))
            confidence.shape = 1, 45390, 21
            location.shape = 1, 45390, 4
            seg_logits.shape = 1, 75, 75, 21

            for cid in np.unique(gt_cats):
                mask = (gt_cats == cid)
                bbox = gt_bboxes[mask]
                diff = difficulty[mask]
                det = np.zeros(len(diff), dtype=np.bool)
                gt[cid][img_id] = {'bbox': bbox, 'difficult': diff, 'det': det}

            confidence1 = confidence
            location1 = location
            seg_logits1 = seg_logits
            output = detector.feed_forward(img, seg_gt, confidence1, location1, seg_logits1,
                                           w, h, name, gt_bboxes, gt_cats) # result

            if args.detect:
                det_bboxes, det_probs, det_cats = output[:3]
                for i in range(len(det_cats)):
                    dets[det_cats[i]].append((img_id, det_probs[i]) + tuple(det_bboxes[i]))

        # print("gt = ", gt)
        # print("dets = ", dets)
        print("table result:")
        table = make_detection_table(gt, dets, loader) if args.detect else None
        print("iou result:")
        iou = compute_mean_iou(detector) if args.segment else None


if __name__ == '__main__':
    tf.app.run()