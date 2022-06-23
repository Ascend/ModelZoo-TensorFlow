from npu_bridge.npu_init import *
import os
import json
import csv
import glob
import argparse
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np


def read_json(data_dir, json_string):
    print('xxx', data_dir)
    json_paths = glob.glob(os.path.join(data_dir, json_string))
    print('sss', json_paths)
    data = []
    for path in json_paths:
        with open(path) as f:
            d = (line.strip() for line in f)
            d_str = "[{0}]".format(','.join(d))
            data.append(json.loads(d_str))

    num_samples = 0
    for d in data:
        num_samples += len(d)
    print('Number of labeled images:', num_samples)
    #print('data keys:', data[0][0].keys())

    return data


def read_image_strings(data, input_dir):
    img_paths = []
    for datum in data:
        for d in datum:
            path = os.path.join(input_dir, d['raw_file'])
            img_paths.append(path)

    num_samples = 0
    for d in data:
        num_samples += len(d)
    assert len(img_paths) == num_samples, 'Number of samples do not match'
    print(img_paths[0:2])

    return img_paths


def save_input_images(output_dir, img_paths):
    output_path2 = os.path.join(output_dir, 'images')
    os.makedirs(output_path2)
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        img = cv2.imread(path)
        output_path = os.path.join(output_path2, '{}.png'.format(str(i).zfill(4)))
        cv2.imwrite(output_path, img)
    #mox.file.copy(output_path, 's3://cann001/dataset/dataset/')


def draw_lines(img, lanes, height, instancewise=False):
    for i, lane in enumerate(lanes):
        pts = [[x, y] for x, y in zip(lane, height) if (x != -2 and y != -2)]
        pts = np.array([pts])
        if not instancewise:
            cv2.polylines(img, pts, False, 255, thickness=7)
        else:
            cv2.polylines(img, pts, False, 50 * i + 20, thickness=7)


def draw_single_line(img, lane, height):
    pts = [[x, y] for x, y in zip(lane, height) if (x != -2 and y != -2)]
    pts = np.array([pts])
    cv2.polylines(img, pts, False, 255, thickness=15)


def save_label_images(output_dir, data, instancewise=True):
    counter = 0
    output_path1 = os.path.join(output_dir, 'labels')
    os.makedirs(output_path1)
    for i in range(len(data)):
        for j in tqdm(range(len(data[i]))):
            img = np.zeros([720, 1280], dtype=np.uint8)
            lanes = data[i][j]['lanes']
            height = data[i][j]['h_samples']
            draw_lines(img, lanes, height, instancewise)
            output_path = os.path.join(output_path1, '{}.png'.format(str(counter).zfill(4)))
            cv2.imwrite(output_path, img)
            counter += 1
    #mox.file.copy(output_path, 's3://cann001/dataset/dataset/')


if __name__ == '__main__':

    print('begin to run')

    import moxing as mox
    parser = argparse.ArgumentParser()
    #parser.add_argument('srcdir',  help="Source directory of TuSimple dataset")
    parser.add_argument('--srcdir', type=str, default='s3://cann001/dataset/train_set', help="Source directory of TuSimple dataset")
    parser.add_argument('--output', type=str, default='../dataset')

    parser.add_argument("--train_url0", type=str, default="s3://cann001/dataset/dataset/")
    local_dir = '/cache/11'
    local_dir2 = '/cache/22'
    os.makedirs(local_dir2)

    #args = parser.parse_args()
    args, unkown = parser.parse_known_args()
    mox.file.copy_parallel(args.srcdir, local_dir)

    print('node2')
    '''
    if not os.path.isdir(args.srcdir):
        raise IOError('Directory does not exist')
    
    if not os.path.isdir('images'):
        os.mkdir('images')
    if not os.path.isdir('labels'):
        os.mkdir('labels')
    '''
    json_string = 'label_data_*.json'
    data = read_json(local_dir, json_string)
    img_paths = read_image_strings(data, local_dir)
    print('dddd', data)
    print('ppppp', img_paths)
    save_input_images(local_dir2, img_paths)
    save_label_images(local_dir2, data)
    p = os.listdir(local_dir2)
    print(p)
    print('Yes')
    '''
    opdir = os.path.join(args.train_url, 'result')
    if not mox.file.exists(opdir):
        mox.file.make_dirs(opdir)
    #opdir = 's3://cann001/dataset/dataset/'
    '''
    #local_dir = os.path.join(local_dir, 'images')
    mox.file.copy_parallel(local_dir2, args.train_url0)
    print('123456')
    #mox.file.copy_parallel(local_dir2, args.train_url0)
