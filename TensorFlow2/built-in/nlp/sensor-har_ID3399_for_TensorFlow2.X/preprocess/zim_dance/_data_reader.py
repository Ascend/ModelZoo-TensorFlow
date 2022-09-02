import csv
import os.path as osp
import yaml

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from rich.console import Console

plt.style.use('ggplot')
CONSOLE = Console()


class data_reader:
    ANN_FILE_PATH = 'data/annotations/zim-dance-valse.txt'
    DATA_PATH = 'data/raw/zim/merged/'
    SENSORS = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']

    def __init__(self, train_test_files, use_columns, output_file_name,
                 verbose, use_length, use_stats):
        self.use_length = use_length
        self.use_stats = use_stats
        self.ind_to_sensor = {i+1: k for i, k in enumerate(self.SENSORS)}

        if not osp.exists(output_file_name):
            self.data, self.idToLabel = self.read_zim(
                train_test_files, use_columns, verbose)
            self.save_data(output_file_name)

    def save_data(self, output_file_name):
        with h5py.File(output_file_name, 'w') as f:
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])

    def normalize(self, x):
        """Min-Max normalization.
           Values taken from data_proc.yaml"""
        # min_val, max_val = -1, 1
        # values taken globally from data_proc.yaml
        min_val, max_val = 71, 179
        return (x - min_val) / (max_val - min_val)

    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    def read_zim(self, train_test_files, use_columns, verbose):
        files = train_test_files
        with open(self.ANN_FILE_PATH, 'r') as ann:
            label_map = []
            for line in ann:
                (id, label) = line.split(' ', 1)
                label_map.append((id, label.strip()))

        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        if verbose:
            CONSOLE.print('\n =====   Label maps   =====\n', style='green')
            print(label_map)
            CONSOLE.print('\n =====   Label to Id   =====\n', style='green')
            print(labelToId)
            CONSOLE.print('\n =====   Id to Label   =====\n', style='green')
            print(idToLabel)

        cols = use_columns
        data = {dataset: self.read_zim_files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data, idToLabel

    def read_zim_files(self, filelist, cols, labelToId):
        data = []
        labels = []
        for _, filename in enumerate(filelist):
            with open(osp.join(self.DATA_PATH, filename), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                former_cls, count = 99, 0
                sensor_data = {k: [] for k in self.SENSORS}

                for line in reader:
                    if (former_cls != line[0]) & (former_cls != 99):
                        # add total length
                        if self.use_length:
                            for i in range(count, 0, -1):
                                data[len(data)-i][6] = float(self.normalize(count)) / 10

                        # CONSOLE.print(f'================================================================ Before', style='bold yellow')
                        # for i in range(count, 0, -1):
                        #     CONSOLE.print(data[len(data)-i])

                        # add statistics
                        if self.use_stats:
                            for i in range(count, 0, -1):
                                if self.use_length:
                                    j, balancer = 7, 6
                                    limit = 19
                                else:
                                    j, balancer = 6, 5
                                    limit = 18
                                while j < limit:
                                    mean = np.mean(
                                        sensor_data[self.ind_to_sensor[j-balancer]]) / 10
                                    std = np.std(
                                        sensor_data[self.ind_to_sensor[j-balancer]]) / 10
                                    # p_25 = np.mean(np.percentile(sensor_data[self.ind_to_sensor[j-balancer]], list(range(23, 28)))) / 10
                                    # mode = stats.mode([round(sd, 2) for sd in sensor_data[self.ind_to_sensor[j-balancer]]])[0][0] / 10
                                    # p_50 = np.mean(np.percentile(sensor_data[self.ind_to_sensor[j-balancer]], list(range(35, 45)))) / 10
                                    # p_75 = np.mean(np.percentile(sensor_data[self.ind_to_sensor[j-balancer]], list(range(65, 75)))) / 10

                                    data[len(data)-i][j] = mean
                                    j += 1
                                    data[len(data)-i][j] = std
                                    j += 1
                                    # data[len(data)-i][j] = p_50
                                    # j += 1
                                    # data[len(data)-i][j] = mode
                                    # j += 1
                                    # data[len(data)-i][j] = p_75
                                    # j += 1
                                    balancer += 1

                        # CONSOLE.print(f'================================================================ After', style='bold yellow')
                        # for i in range(count, 0, -1):
                        #     CONSOLE.print(data[len(data)-i])
                        count = 1
                        sensor_data = {k: [] for k in self.SENSORS}
                    else:
                        count += 1
                    former_cls = line[0]

                    elem = []
                    for ind in cols:
                        # sample sensor data for statistic calculations
                        if (ind > 0) and (ind < 7):
                            sensor_data[self.ind_to_sensor[ind]].append(float(line[ind]))

                        if (ind == 7) and (self.use_length):
                            elem.append(self.normalize(count) / 10)
                        elif (ind >= 7) and (self.use_stats):
                            # ! adding dummy values for stat features
                            elem.append(self.normalize(count) / 10)
                        else:
                            # append sensor data as is
                            elem.append(line[ind])

                    data.append([float(x) / 10 for x in elem[1:]])
                    labels.append(labelToId[elem[0]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


def read_dataset(train_test_files, use_columns, output_file_name, verbose,
                 use_length=False, use_stats=False):
    CONSOLE.print('[Reading ZIM] ...', style='bold green')
    data_reader(train_test_files, use_columns, output_file_name, verbose,
                use_length, use_stats)
    CONSOLE.print('[Reading ZIM] : DONE', style='bold green')
