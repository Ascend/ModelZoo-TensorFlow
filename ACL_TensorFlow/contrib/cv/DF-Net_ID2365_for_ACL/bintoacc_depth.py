#加载所有bin文件为np.array,使用eval_depth.py

from __future__ import division
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

from kitti_eval.depth_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, default='E:/datasets/DF-Net_dataset/KITTI/raw/data/', help='Path to the KITTI dataset directory')
parser.add_argument("--pred_bin_file", type=str, default='./depth_output/', help="Path to the prediction file")
parser.add_argument("--split", type=str, default='test')
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def main():
    if args.split == 'val':
        test_file_list = './data/kitti/mytest_files_eigen.txt'
    elif args.split == 'test':
        test_file_list = './data/kitti/mytest_files_eigen.txt'
    else:
        assert False

    # 读取bin文件 eg ./depth_output/0002/00000000000_output_0.bin
    with open(test_file_list, 'r') as f:
        test_files = f.readlines()
        test_files = [args.pred_bin_file + (t[28:32] + '/' + t[52:62]) for t in test_files]
    pred_depths = np.zeros([121, 160, 576])
    i = 0
    for test_file in test_files:
        pred_depth = np.fromfile(test_file+'_output_0.bin', dtype=np.float32)
        pred_depths[i] = pred_depth.reshape([160, 576])
        #后处理
        pred_depths[i] = [1. / disp for disp in pred_depths[i]]
        i += 1

    
    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(read_text_lines(test_file_list), args.kitti_dir)
    num_test = len(im_files)
    gt_depths = []
    pred_depths_resized = []
    invalid_ids = []
    for t_id in range(num_test):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        # Some frames in val set do not have ground truth labels
        try:
            depth = generate_depth_map(gt_calib[t_id],
                                       gt_files[t_id],
                                       im_sizes[t_id],
                                       camera_id,
                                       False,
                                       True)
            gt_depths.append(depth.astype(np.float32))

            pred_depths_resized.append(
                cv2.resize(pred_depths[t_id],
                           (im_sizes[t_id][1], im_sizes[t_id][0]),
                           interpolation=cv2.INTER_LINEAR))
        except:
            invalid_ids.append(t_id)
            print(t_id)
    pred_depths = pred_depths_resized
    num_test -= len(invalid_ids)

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)

    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > args.min_depth,
                              gt_depth < args.max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

main()
