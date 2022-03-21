import argparse
import os
import pathlib
import sys
import time
sys.path.append("../")
import cv2
import numpy as np
import tensorflow as tf
from absl import logging

from modules.retina_slim import RetinaFaceModel, test_gn
from modules.utils import (draw_bbox_landm, load_info, load_sess, load_yaml,
                           pad_input_image, recover_pad_output,
                           set_memory_growth)


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")

class Configs(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Global set
        parser.add_argument('--model', type=str, default='res50')
        parser.add_argument('--gpu', type=eval, default=(6,7))
        parser.add_argument('--log_level', type=str, default='3')
        parser.add_argument('--save_path', type=str,
                            default='widerface_evaluate/widerface_txt/')
        parser.add_argument('--origin_size', type=str2bool, default=True)
        parser.add_argument('--save_image', type=str2bool, default=True)
        parser.add_argument('--iou_th', type=float, default=0.4)
        parser.add_argument('--score_th', type=float, default=0.02)
        # will be changed
        parser.add_argument('--vis_th', type=float, default=0.0)
        self.args = parser.parse_args()


def main(args):
    root_path, _ = os.path.split(os.path.abspath(__file__))
    ######################## init ########################
    res_config = os.path.join(root_path, "configs/retinaface_res50.yaml")
    mbn_config = os.path.join(root_path, "configs/retinaface_mbv2.yaml")
    config_path = res_config if args.model == 'res50' else mbn_config
    visible_devices = ''
    for i in args.gpu:
        visible_devices += '{}, '.format(i)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.log_level
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    # set log
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    # set_memory_growth()

    # load config
    cfg = load_yaml(config_path)
    cfg.update({'root_path': root_path})

    # set checkpoint path
    checkpoint_dir = os.path.join(root_path, 'checkpoints', cfg['sub_name'])
    slide_steps = cfg['steps']
    base_anchor_size = cfg['basesize']
    _base_sacle = 2.0**(1.0 / 3)
    five_step_min_sizes = [
        [base_anchor_size*(_base_sacle**(i+j*3)) for i in range(3)] for j in range(len(slide_steps))]
    cfg['min_sizes'] = cfg['min_sizes'] if len(
        slide_steps) == 3 else five_step_min_sizes

    # define network
    model = RetinaFaceModel(cfg, training=False)

    # evaluation on testing dataset
    testset_folder = os.path.join(root_path, cfg['testing_dataset_path'])
    testset_list = os.path.join(testset_folder, 'label.txt')
    img_paths, _ = load_info(testset_list)
    batch_size = cfg['batch_size']

    ######################## build graph ########################
    input_holder = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    # input_holder = tf.placeholder(tf.float32, [1, None, 1024, 3])
    feature = model(input_holder)
    ops = tf.get_default_graph().get_operations()
    _ = [
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, x) for x in ops
        if ("AssignMovingAvg" in x.name and x.type == "AssignSub")
    ]
    upds = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    ######################## session test ########################
    with tf.Session() as sess:
        load_sess(checkpoint_dir, sess=sess, init=False)
        # get test image
        for img_index, img_path in enumerate(img_paths):
            print(" [{} / {}] det {}".format(img_index + 1, len(img_paths), img_path))
            img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_height_raw, img_width_raw, _ = img_raw.shape
            img = np.float32(img_raw.copy())

            # testing scale
            target_size = 1600
            max_size = 2150
            img_shape = img.shape
            img_size_min = np.min(img_shape[0:2])
            img_size_max = np.max(img_shape[0:2])
            resize = float(target_size) / float(img_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * img_size_max) > max_size:
                resize = float(max_size) / float(img_size_max)
            if args.origin_size:
                if os.path.basename(img_path) == '6_Funeral_Funeral_6_618.jpg':
                    resize = 0.5 # this image is too big to avoid OOM problem
                else:
                    resize = 1

            img = cv2.resize(img, None, None, fx=resize, fy=resize,
                            interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
            # in order to fit the gn algrithom
            batch_imgs = [img for _ in range(batch_size)]
            # print(batch_imgs[0].shape)

            # run model
            # outputs = sess.run(feature, feed_dict={input_holder: img[np.newaxis, ...]})
            outputs = sess.run(feature, feed_dict={input_holder: batch_imgs})

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # write results
            img_name = os.path.basename(img_path)
            sub_dir = os.path.basename(os.path.dirname(img_path))
            save_name = os.path.join(
                root_path, args.save_path, sub_dir, img_name.replace('.jpg', '.txt'))

            pathlib.Path(os.path.join(args.save_path, sub_dir)).mkdir(
                parents=True, exist_ok=True)

            with open(save_name, "w") as file:
                bboxs = outputs[:, :4]
                confs = outputs[:, -1]

                file_name = img_name + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                file.write(file_name)
                file.write(bboxs_num)
                for box, conf in zip(bboxs, confs):
                    x = int(box[0] * img_width_raw)
                    y = int(box[1] * img_height_raw)
                    w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                    h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                    confidence = str(conf)
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) \
                        + " " + confidence + " \n"
                    file.write(line)
                print('Have Bbox: '+str(bboxs_num))

            # save images
            pathlib.Path(os.path.join(
                root_path, 'results', cfg['sub_name'], sub_dir)).mkdir(
                    parents=True, exist_ok=True)
            if args.save_image:
                bbox_num = 0
                for prior_index in range(len(outputs)):
                    if outputs[prior_index][15] >= args.vis_th:
                        bbox_num += 1
                        draw_bbox_landm(img_raw, outputs[prior_index],
                                        img_height_raw, img_width_raw)
                cv2.imwrite(os.path.join(root_path, 'results', cfg['sub_name'], sub_dir,
                                        img_name), img_raw)


if __name__ == '__main__':
    all_configs = Configs()
    main(all_configs.args)
