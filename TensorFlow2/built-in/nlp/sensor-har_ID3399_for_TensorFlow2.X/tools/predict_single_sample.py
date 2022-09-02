import json
import yaml
import os
import os.path as osp

import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import accuracy_score
from rich.console import Console

CONSOLE = Console()

def segment_window_all(x_train, y_train, window_size, n_sensor_val):
    window_segments = np.zeros((len(x_train), window_size, n_sensor_val))
    labels = np.zeros((len(y_train),))

    total_len = len(x_train)

    for i in range(total_len):
        end = i + window_size

        if end > total_len:
            pad_len = end - total_len
            window_segments[i] = x_train[i - pad_len:end]
            labels[i] = y_train[total_len - 1]
        else:
            window_segments[i] = x_train[i:end]
            labels[i] = y_train[end - 1]

    return window_segments, labels


def clean(str):
    return str.replace('_', ' ').lower()


def normalize(x):
    """Min-Max normalization.
        Values taken from data_proc.yaml"""
    min_val, max_val = 71, 179
    return (x - min_val) / (max_val - min_val)


def parse_args():
    parser = ArgumentParser(prog='get prediction for single sample')
    parser.add_argument(
        'path',
        type=str,
        help='path to sample or dir')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='saved_model/zim',
        help='model dir')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-valse.txt',
        help='annotation file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/raw/zim/single_test',
        help='out dir to save processed sample')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    config_file = open('configs/data.yaml', 'r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config['zim']
    model_file = open('configs/model.yaml', 'r')
    model_config = yaml.load(model_file, Loader=yaml.FullLoader)
    model = tf.keras.models.load_model(args.model_dir)

    # data pre-processing
    label_to_number = {}
    with open(args.ann, 'r') as ann:
        for line in ann:
            (val, key) = line.split(' ', 1)
            label_to_number[key.strip()] = int(val)

    to_process = []
    if osp.isdir(args.path):
        to_process += os.listdir(args.path)
    else:
        to_process.append(args.path.split('/')[-1])
        args.path = '/'.join(p for p in args.path.split('/')[:-1])

    sensors = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
    ind_to_sensor = {i: sensor for i, sensor in enumerate(sensors)}

    for sample in to_process:
        if not sample.endswith('.json'):
            continue

        sensor_data = {k: [] for k in sensors}
        label = clean(sample.split('2021')[0])
        activity_id = label_to_number.get(label.strip(), None)
        x_test = []
        y_test = []

        content = open(osp.join(args.path, sample), 'r')
        content = json.load(content)

        for row in content:
            for i in range(len(sensors)):
                sensor_data[ind_to_sensor[i]].append(row[i])

        stat_data = []
        for sensor in sensors:
            stat_data.append(np.mean(sensor_data[sensor]))
            stat_data.append(np.std(sensor_data[sensor]))
            # stat_data.append(np.mean(
            #     np.percentile(sensor_data[sensor], list(range(35, 45)))))
            # stat_data.append(np.mean(
            #     np.percentile(sensor_data[sensor], list(range(65, 75)))))

        for row in content:
            result = [activity_id]
            result.extend([x for x in row])
            result.append(normalize(len(content)) / 10)
            # for st in stat_data:
            #     result.append(st)
            x_test.append([float(x) / 10 for x in result[1:]])
            y_test.append(result[0])

        n_sensor_val = len(config['feature_columns']) - 1
        # replace any nan with mean
        x_test = np.where(
            np.isnan(x_test),
            np.ma.array(x_test, mask=np.isnan(x_test)).mean(axis=0), x_test)

        # window
        try:
            test_x, test_y = segment_window_all(x_test, y_test, config['window_size'], n_sensor_val)
            test_y = tf.keras.utils.to_categorical(test_y)
        except ValueError as e:
            CONSOLE.print(f'Error {e}. Could not predict sample {sample}', style='red')
            continue

        # predict
        pred = model.predict(
            test_x,
            batch_size=model_config['zim']['batch_size'],
            verbose=1)
        acc = accuracy_score(
            np.argmax(test_y, axis=1),
            np.argmax(pred, axis=1),
            normalize=True)
        # print(test_y)
        # print(np.argmax(test_y, axis=1))
        # print()
        # print(pred)
        # print(np.argmax(pred, axis=1))
        CONSOLE.print(sample, style='green')
        CONSOLE.print(f'The model is {round(100*acc, 2)}% confident '
                      f'that the sample {sample} is {label}', style='green')


if __name__ == '__main__':
    main()
