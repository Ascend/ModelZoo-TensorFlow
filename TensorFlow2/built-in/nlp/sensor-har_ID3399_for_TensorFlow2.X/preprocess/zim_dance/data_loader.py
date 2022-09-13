import os
import os.path as osp

import h5py
import numpy as np
import tensorflow as tf
import yaml

from pathlib import Path
from ._data_reader import read_dataset
from ._sliding_window import segment_pa2, segment_window_all
from rich.console import Console


CONSOLE = Console()


def get_zim_data():
    config_file = open('configs/data.yaml', mode='r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config['zim']
    Path(data_config['data_dir']['processed']).mkdir(
                        parents=True, exist_ok=True)

    train_test_files = {'train': config['train_files'],
                        'validation': config['validation_files'],
                        'test': config['test_files']}

    read_dataset(train_test_files=train_test_files,
                 use_columns=config['feature_columns'],
                 output_file_name=osp.join(
                    data_config['data_dir']['processed'],
                    config['output_file']),
                 verbose=config['verbose'],
                 use_length=config['use_length'],
                 use_stats=config['use_stats'])

    path = osp.join(
        data_config['data_dir']['processed'],
        config['output_file'])
    f = h5py.File(path, 'r')

    x_train = f.get('train').get('inputs')[()]
    y_train = f.get('train').get('targets')[()]

    x_val = f.get('validation').get('inputs')[()]
    y_val = f.get('validation').get('targets')[()]

    x_test = f.get('test').get('inputs')[()]
    y_test = f.get('test').get('targets')[()]

    if config['verbose']:
        CONSOLE.print('=====\n\n', style='bold green')
        print("x_train shape = ", x_train.shape)
        print("y_train shape =", y_train.shape)

        print("x_val shape = ", x_val.shape)
        print("y_val shape =", y_val.shape)

        print("x_test shape =", x_test.shape)
        print("y_test shape =", y_test.shape)
        CONSOLE.print('\n\n=====', style='bold green')

    # replace nan with mean
    x_train = np.where(
        np.isnan(x_train),
        np.ma.array(x_train, mask=np.isnan(x_train)).mean(axis=0), x_train)
    x_val = np.where(
        np.isnan(x_val),
        np.ma.array(x_val, mask=np.isnan(x_val)).mean(axis=0), x_val)
    x_test = np.where(
        np.isnan(x_test),
        np.ma.array(x_test, mask=np.isnan(x_test)).mean(axis=0), x_test)

    n_sensor_val = len(config['feature_columns']) - 1
    train_x, train_y = segment_pa2(x_train, y_train, config['window_size'], n_sensor_val)
    val_x, val_y = segment_pa2(x_val, y_val, config['window_size'], n_sensor_val)
    test_x, test_y = segment_window_all(x_test, y_test, config['window_size'], n_sensor_val)

    if config['verbose']:
        CONSOLE.print('=====\n\n', style='bold green')
        print(f'Number of sensors: {n_sensor_val}')

        print("train_x shape =", train_x.shape)
        print("train_y shape =", train_y.shape)
        print('train_y distribution', np.unique(train_y, return_counts=True))

        print("val_x shape =", val_x.shape)
        print("val_y shape =", val_y.shape)
        print('val_y distribution', np.unique(val_y, return_counts=True))

        print("test_x shape =", test_x.shape)
        print("test_y shape =", test_y.shape)
        print('test_y distribution', np.unique(test_y, return_counts=True))
        CONSOLE.print('\n\n=====', style='bold green')

    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    val_y = tf.keras.utils.to_categorical(val_y)

    if config['verbose']:
        CONSOLE.print('=====\n\n', style='bold green')
        print("unique test_y", np.unique(test_y))
        print("unique train_y", np.unique(train_y))
        print("test_y[1]=", test_y[1])

        print("train_y shape(1-hot) =", train_y.shape)
        print("val_y shape(1-hot) =", val_y.shape)
        print("test_y shape(1-hot) =", test_y.shape)
        CONSOLE.print('\n\n=====', style='bold green')

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
