import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


class data_reader:
    def __init__(self, train_test_files, use_columns, output_file_name, verbose, data_path):
        self.data_path = data_path
        if not os.path.exists(output_file_name):
            self.data, self.idToLabel = self.readPamap2(train_test_files, use_columns, verbose)
            self.save_data(output_file_name)

    def save_data(self, output_file_name):
        with h5py.File(output_file_name, 'w') as f:
            for key in self.data:
                # groups are folder-like containers that hold datasets
                # and other groups; work like dictionaries
                f.create_group(key)
                for field in self.data[key]:
                    # datsets are array-like collections of data
                    # work like numpy arrays
                    f[key].create_dataset(field, data=self.data[key][field])

    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    def readPamap2(self, train_test_files, use_columns, verbose):
        files = train_test_files
        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        if verbose:
            print('\n =====   Label maps   =====\n')
            print(label_map)
            print('\n =====   Label to Id   =====\n')
            print(labelToId)
            print('\n =====   Id to Label   =====\n')
            print(idToLabel)

        cols = use_columns
        data = {dataset: self.readPamap2Files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data, idToLabel

    def readPamap2Files(self, filelist, cols, labelToId):
        """
        filelist: { train: [ 'subject101.dat', ..., 'subject109.dat' ] }
        cols: [ 1, 4, 5, 6, ..., 40, 44, 45, 46 ]
        labelsToId: { '1': 0, '2': 1, '3': 2, '4': 3, ...
                    '19': 15, '20': 16, '24': 17 }
        """
        data = []
        labels = []
        data_path = os.path.join(self.data_path, "data/raw/pamap2/PAMAP2_Dataset/Protocol") 
        for i, filename in enumerate(filelist):
            # print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{data_path}/{filename}', 'r') as f:
            # with open('data/raw/pamap2/PAMAP2_Dataset/Protocol/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the non related activity
                    if line[1] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) < 9:
                        # originally was for x in elem[:-1]
                        # see https://github.com/saif-mahmud/self-attention-HAR/issues/3
                        data.append([float(x) / 1000 for x in elem[1:]])
                        labels.append(labelToId[elem[0]])

        # return the list of record + their corresponding labels
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int) + 1}


def read_dataset(train_test_files, use_columns, output_file_name, verbose, data_path):
    print('[Reading PAMAP2] ...')
    data_reader(train_test_files, use_columns, output_file_name, verbose, data_path)
    print('[Reading PAMAP2] : DONE')
