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
#


""" dataset.py """

import os
from collections import Counter
import numpy as np
import scipy.sparse as sp

class CoraData():
    """ cora dataset """
    def __init__(self, data_path='data/', cora_full=False, shuffle=False, take_subgraphs=False,
                 train_size=None, valid_size=500, test_size=1000,
                 min_train_samples_per_class=0, min_valid_samples_per_class=0, save_inputs=False,
                 out_path='result/'):
        self.cora_full = cora_full
        self.data_path = os.path.join(data_path, 'cora_full.npz') if cora_full \
            else os.path.join(data_path, 'cora.npz')
        self.features, self.labels, self.adjacency, self.classes, self.num_nodes, \
            self.num_classes, self.feature_dim = self.load_data()
        min_samples_per_class = min_train_samples_per_class + min_valid_samples_per_class
        self.standadize(take_subgraphs, min_samples_per_class, n_components=1)
        self.normalize_features()
        self.normalize_adjacency()
        self.num_edges = (self.adjacency.count_nonzero() - self.num_nodes) // 2
        if shuffle:
            self.shuffle_nodes()
        self.train_mask, self.valid_mask, self.test_mask = None, None, None
        self.split_data(train_size, valid_size, test_size,
                        min_train_samples_per_class, min_valid_samples_per_class)
        if save_inputs:
            # np.save(os.path.join(data_path, "features.npy"), self.features)
            # np.save(os.path.join(data_path, "labels.npy"), self.labels)
            # np.save(os.path.join(data_path, "adjacency.npy"), self.adjacency.todense())
            # np.save(os.path.join(data_path, "mask.npy"), self.test_mask)
            np.save(os.path.join(out_path, "features.npy"), self.features)
            np.save(os.path.join(out_path, "labels.npy"), self.labels)
            np.save(os.path.join(out_path, "adjacency.npy"), self.adjacency.todense())
            np.save(os.path.join(out_path, "mask.npy"), self.test_mask)

    def load_data(self):
        """ load dataset """
        print("Processing Cora dataset ...")
        with np.load(self.data_path) as f:  # pylint: disable=invalid-name
            features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                                     shape=f['attr_shape'], dtype=np.float32)
            labels = f['labels'].astype(np.int32)
            adjacency = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                                      shape=f['adj_shape'], dtype=np.float32)
            features.data[:] = 1
            adjacency.data[:] = 1
            adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency)
            num_nodes, feature_dim = features.shape
            classes = list(np.unique(labels))
            num_classes = len(classes)
        return features, labels, adjacency, classes, num_nodes, num_classes, feature_dim

    def standadize(self, take_subgraphs=False, min_samples_per_class=0, n_components=1):
        """ standadize graph data """
        if take_subgraphs:
            self.take_largest_subgraphs(n_components)
        if min_samples_per_class > 0:
            self.remove_underrepresented_classes(min_samples_per_class)
        labels_to_keep = np.unique(self.labels)
        if len(labels_to_keep) < self.num_classes:
            self.relabel(labels_to_keep)

    def relabel(self, labels_to_keep=None):
        """ remove unused labels """
        if labels_to_keep is None:
            labels_to_keep = np.unique(self.labels)
        self.labels = np.array([i for j in range(self.num_nodes)
                                for i, l in enumerate(labels_to_keep)
                                if self.labels[j] == l], dtype=np.int32)
        self.classes = list(np.unique(self.labels))
        self.num_classes = len(self.classes)

    def remove_underrepresented_classes(self, min_samples_per_class=0):
        """ remove underrepresented classes """
        label_counter = Counter(self.labels)
        labels_to_keep = set(l for l, n in label_counter.items() if n >= min_samples_per_class)
        nodes_to_keep = [i for i in range(self.num_nodes) if self.labels[i] in labels_to_keep]
        self.features = self.features[nodes_to_keep]
        self.adjacency = self.adjacency[nodes_to_keep][:, nodes_to_keep]
        self.labels = self.labels[nodes_to_keep]
        self.num_nodes = len(nodes_to_keep)

    def take_largest_subgraphs(self, n_components=1):
        """ take largest subgraphs """
        _, component_idx = sp.csgraph.connected_components(self.adjacency, directed=False)
        component_sizes = np.bincount(component_idx)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
        nodes_to_keep = [i for i, c in enumerate(component_idx) if c in components_to_keep]
        self.features = self.features[nodes_to_keep]
        self.adjacency = self.adjacency[nodes_to_keep][:, nodes_to_keep]
        self.labels = self.labels[nodes_to_keep]
        self.num_nodes = len(nodes_to_keep)

    def normalize_adjacency(self):
        """ normalize adjacency matrix """
        adj = self.adjacency + sp.eye(self.num_nodes, dtype=np.float32)
        deg = np.array(adj.sum(1))
        d_hat = sp.diags(np.power(deg, -0.5).flatten())
        self.adjacency = d_hat.dot(adj).dot(d_hat)

    def normalize_features(self):
        """ normalize features """
        self.features = self.features / self.features.sum(1)

    def shuffle_nodes(self):
        """ shuffle nodes """
        idx_map = np.random.permutation(self.num_nodes)
        self.features = self.features[idx_map]
        self.labels = self.labels[idx_map]
        self.adjacency = self.adjacency[idx_map][:, idx_map]

    def split_data(self, train_size, valid_size, test_size,
                   min_train_samples_per_class, min_valid_samples_per_class):
        """ split dataset """
        if train_size is None:
            train_size = min_train_samples_per_class * self.num_classes
        if valid_size is None:
            valid_size = min_valid_samples_per_class * self.num_classes

        train_mask, valid_mask, test_mask = [], [], []
        for label in self.classes:
            idx_ = list(np.where(self.labels==label)[0])
            train_mask += idx_[:min_train_samples_per_class]
            valid_mask += idx_[min_train_samples_per_class:
                               min_train_samples_per_class + min_valid_samples_per_class]
        idx_used = train_mask + valid_mask
        train_mask_counter = train_size - len(train_mask)
        valid_mask_counter = valid_size - len(valid_mask)
        test_mask_counter = test_size
        for i in range(self.num_nodes):
            if train_mask_counter > 0:
                if i not in idx_used:
                    train_mask.append(i)
                    train_mask_counter -= 1
            elif valid_mask_counter > 0:
                if i not in idx_used:
                    valid_mask.append(i)
                    valid_mask_counter -= 1
            elif test_mask_counter > 0:
                if i not in idx_used:
                    test_mask.append(i)
                    test_mask_counter -= 1
            else:
                break
        self.train_mask = np.array(sorted(train_mask), dtype=np.int32)
        self.valid_mask = np.array(sorted(valid_mask), dtype=np.int32)
        self.test_mask = np.array(sorted(test_mask), dtype=np.int32)
