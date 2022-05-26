from __future__ import division
import cv2
import numpy as np
import os
import png
import time

from core import flow_to_image, write_flow

batch_size = 1
img_height = 384
img_width = 1280
dataset_dir = 'E:/datasets/DF-Net_dataset/data_scene_flow_2015/training/'
output_dir = './flow_output/'
pyr_lvls = 6
flow_pred_lvl = 2


# kitti 2012 has 200 training pairs, 200 test pairs
NUM = 200


def postproc_y_hat_test(y_hat, adapt_info=None):
    """Postprocess the results coming from the network during the test mode.
    Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
    Args:
        y_hat: predictions, see set_output_tnsrs() for details
        adapt_info: adaptation information in (N,H,W,2) format
    Returns:
        Postprocessed labels
    """
    #assert (isinstance(y_hat, list) and len(y_hat) == 2)

    # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
    pred_flows = y_hat
    if adapt_info is not None:
        pred_flows = pred_flows[:, 0:adapt_info[0], 0:adapt_info[1], :]

    return pred_flows

def get_flow(path):
    bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    invalid = bgr[:, :, 0] == 0
    out_flow = (bgr[:, :, 2:0:-1].astype('f4') - 2 ** 15) / 64.
    out_flow[invalid] = 0
    return out_flow, bgr[:, :, 0]


def compute_flow_error(gt_flow, pred_flow, mask):
    H, W, _ = gt_flow.shape
    old_H, old_W, _ = pred_flow.shape

    # Reshape predicted flow to have same size as ground truth
    pred0 = cv2.resize(pred_flow[:, :, 0], (W, H), interpolation=cv2.INTER_LINEAR) * (1.0 * W / old_W)
    pred1 = cv2.resize(pred_flow[:, :, 1], (W, H), interpolation=cv2.INTER_LINEAR) * (1.0 * H / old_H)
    pred = np.stack((pred0, pred1), axis=-1)

    err = np.sqrt(np.sum(np.square(gt_flow - pred), axis=-1))
    err_valid = np.sum(err * mask) / np.sum(mask)

    gt_mag = np.sqrt(np.sum(np.square(gt_flow), axis=-1))
    mask1 = err > 3.
    mask2 = err / (gt_mag + 1e-12) > 0.05
    final_mask = np.logical_and(np.logical_and(mask1, mask2), mask)
    f1 = final_mask.sum() / mask.sum()

    return err_valid, pred, f1


def compute_pyr_error(gt_flow, pred_flow, pred_shape, mask):  # pred_flow: [96, 360, 2]
    # by reeki
    H, W, _ = gt_flow.shape
    old_H, old_W, _ = pred_flow.shape
    flow_H, flow_W = pred_shape[0], pred_shape[1]

    # Reshape predicted pyr to pred_flow becase flownet dose
    flow0 = cv2.resize(pred_flow[:, :, 0], (flow_W, flow_H), interpolation=cv2.INTER_LINEAR) * (1.0 * flow_W / old_W)
    flow1 = cv2.resize(pred_flow[:, :, 1], (flow_W, flow_H), interpolation=cv2.INTER_LINEAR) * (1.0 * flow_H / old_H)

    # flow_pred = tf.image.resize_bilinear(pred_flow, (flow_H, flow_W), name="flow_pred") * FLOW_SCALE #why different?

    # Reshape predicted flow to have same size as x_adapt.shape
    pred0 = flow0[0:H, 0:W]
    pred1 = flow1[0:H, 0:W]
    pred = np.stack((pred0, pred1), axis=-1)

    # pred = flow_pred[0,0:H, 0:W, :]

    err = np.sqrt(np.sum(np.square(gt_flow - pred), axis=-1))
    err_valid = np.sum(err * mask) / np.sum(mask)

    gt_mag = np.sqrt(np.sum(np.square(gt_flow), axis=-1))
    mask1 = err > 3.
    mask2 = err / (gt_mag + 1e-12) > 0.05
    final_mask = np.logical_and(np.logical_and(mask1, mask2), mask)
    f1 = final_mask.sum() / mask.sum()

    return err_valid, pred, f1


def write_flow_png(name, flow):
    H, W, _ = flow.shape
    out = np.ones((H, W, 3), dtype=np.uint64)
    out[:, :, 1] = np.minimum(np.maximum(flow[:, :, 1] * 64. + 2 ** 15, 0), 2 ** 16).astype(np.uint64)
    out[:, :, 0] = np.minimum(np.maximum(flow[:, :, 0] * 64. + 2 ** 15, 0), 2 ** 16).astype(np.uint64)
    with open(name, 'wb') as f:
        writer = png.Writer(width=W, height=H, bitdepth=16)
        im2list = out.reshape(-1, out.shape[1] * out.shape[2]).tolist()
        writer.write(f, im2list)

def pick_frame(path):
    new_files = []
    for i in range(NUM):
        frame1 = os.path.join(path, 'image_2', '{:06d}'.format(i) + '_10.png')
        frame2 = os.path.join(path, 'image_2', '{:06d}'.format(i) + '_11.png')
        new_files.append([frame1, frame2])
    return new_files

def main():
    errs = np.zeros(NUM)
    f1 = np.zeros(NUM)
    tim = np.zeros(NUM)

    if not output_dir is None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(0, NUM):
        if t % 100 == 0:
            print('processing %06d' % t)

        time1 = time.time()
        # 读取bin格式的输出数据
        pred_flows = np.fromfile(output_dir +'2022224_19_29_35_617021/' + '%06d_output_0.bin' % t, dtype=np.float32)
        pred_flows = pred_flows.reshape([1, img_height, img_width, 2])

        # 后处理
        new_files = pick_frame(dataset_dir)
        raw_im0 = cv2.imread(new_files[t][0])
        y_adapt_info = raw_im0.shape # 实际的比例
        # pred_flows_val: (1, 384, 1280, 2)  len(pred_pyrs_val)=5 (1, 6, 20, 2)(1, 12, 40, 2)(1, 24, 80, 2)(1, 48, 160, 2)(1, 96, 320, 2)
        pred_flow_val = postproc_y_hat_test(pred_flows, y_adapt_info)
        # pred_flow_val:  (1, 375, 1242, 2)  len(pred_pyr_val)=1 len(pred_pyr_val[-1])=5  pred_pyr_val[0][-1]: (96, 320, 2)
        tim[t] = time.time() - time1

        # 计算精度
        if 'train' in dataset_dir:
            # no occlusion
            # gt_flow, mask = get_flow(new_files[t][0].replace('image_2', 'flow_noc'))
            # all
            gt_flow, mask = get_flow(new_files[t][0].replace('image_2', 'flow_occ'))
            ### use pred_flows
            errs[t], scaled_pred, f1[t] = compute_flow_error(gt_flow, pred_flow_val[0, :, :, :], mask)
            # pred_flow_val.shape: (1, 96, 320, 2) scaled_pred.shape: (375, 1242, 2) gt_flow.shape: (375, 1242, 2)

        # Save for eval

        # Save for visual colormap
        if not 'test' in dataset_dir and not output_dir is None:
            flow_im = flow_to_image(scaled_pred)  # flow_im: (375, 1242, 3)
            png_name = os.path.join(output_dir, new_files[t][0].split('/')[-1]).replace('png', 'jpg')
            cv2.imwrite(png_name, flow_im[:, :, ::-1])
            # print("flow_im: ", flow_im.shape) #
            flo_name = os.path.join(output_dir, new_files[t][0].split('/')[-1]).replace('png', 'flo')
            write_flow(scaled_pred, flo_name)
        # print(errs[t], f1[t])

    print('{:>10}, {:>10}, {:>10}, {:>10}'.format('(valid) endpoint error', 'f1 score', 'sum time', 'avg time'))
    print('{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}'.format(errs.mean(), f1.mean(), tim.sum(), tim.mean()))


if __name__ == '__main__':
    main()
