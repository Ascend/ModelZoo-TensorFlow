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
# Author: Salli Moustafa (salli.moustafa@huawei.com)
"""
Inference logic for the GCN
"""

from enum import Enum
import numpy as np
import scipy.sparse as sp

import acl

from ats.abstract_task import AbstractTask
from ats.post_processing_manager import PostProcessingManager
from ats.utils import check_status


class InputTypes(Enum):
    """
    Input types
    """
    DATA = 0
    IDX = 1


class SparseInputIds(Enum):
    """
    GCN Sparse Input IDs
    """
    FEATURES_DATA = 0
    FEATURES_IDX = 1
    ADJACENCY_DATA = 2
    ADJACENCY_IDX = 3
    MASK = 4


class DenseInputIds(Enum):
    """
    GCN Dense Input IDs
    """
    FEATURES_DATA = 0
    ADJACENCY_DATA = 1
    MASK = 2


class GCNTask(AbstractTask):
    """
    Implements the core logic for the GCN inference as expected by ATS
    """

    def __init__(self, dataset, input_path, output_path, sparse):
        self.dataset = dataset
        self.input_path = input_path
        self.output_path = output_path
        self.sparse = sparse
        self.input_ids = SparseInputIds if self.sparse else DenseInputIds

        self.features, self.adjacency, self.labels, self.mask = GCNTask.__load_inputs(input_path,
                                                                                      sparse)

        self.post_processing_manager = PostProcessingManager(1, 100)

    def init(self, device_id, acl_resource):
        """
        Instantiates the model on the current device
        """

        self.post_processing_manager.init(device_id)

        self.dataset.create(device_id, acl_resource.stream_count)

        assert all(self.dataset.get_input_dims(device_id,
                                               self.input_ids.FEATURES_DATA.value)['dims']
                   ==
                   np.array(self.features[int(InputTypes.DATA.value)].shape)), \
            "Input features shape does not match model requirements"
        assert all(self.dataset.get_input_dims(device_id,
                                               self.input_ids.ADJACENCY_DATA.value)['dims']
                   ==
                   np.array(self.adjacency[int(InputTypes.DATA.value)].shape)), \
            "Input adjacency shape does not match model requirements"

    def finalize(self, device_id, stream_count):
        """
        Finalizes the model and free acquired resources
        """

        self.dataset.destroy(device_id, stream_count)

        self.post_processing_manager.finalize(device_id)

    def run(self, device_id, acl_resource):
        """
        GCN inference workflow
        """

        stream_id = 0

        self.post_processing_manager.subscribe(device_id,
                                               acl_resource.context,
                                               acl_resource.streams)

        features_data, features_data_size = self.__get_input_data(self.input_ids.FEATURES_DATA)
        if self.sparse:
            features_idx, features_idx_size = self.__get_input_data(self.input_ids.FEATURES_IDX)
        adjacency_data, adjacency_data_size = self.__get_input_data(self.input_ids.ADJACENCY_DATA)
        if self.sparse:
            adjacency_idx, adjacency_idx_size = self.__get_input_data(self.input_ids.ADJACENCY_IDX)
        mask, mask_size = self.__get_input_data(self.input_ids.MASK)

        status = self.dataset.push_async(features_data,
                                         self.input_ids.FEATURES_DATA.value,
                                         features_data_size,
                                         acl_resource.run_mode,
                                         device_id,
                                         stream_id,
                                         acl_resource.streams[stream_id])
        check_status("[GCNTask][run] Failed to push features data to the device.", status)

        if self.sparse:
            status = self.dataset.push_async(features_idx,
                                             self.input_ids.FEATURES_IDX.value,
                                             features_idx_size,
                                             acl_resource.run_mode,
                                             device_id,
                                             stream_id,
                                             acl_resource.streams[stream_id])
            check_status("[GCNTask][run] Failed to push features idx to the device.", status)

        status = self.dataset.push_async(adjacency_data,
                                         self.input_ids.ADJACENCY_DATA.value,
                                         adjacency_data_size,
                                         acl_resource.run_mode,
                                         device_id,
                                         stream_id,
                                         acl_resource.streams[stream_id])
        check_status("[GCNTask][run] Failed to push adjacency data to the device.", status)

        if self.sparse:
            status = self.dataset.push_async(adjacency_idx,
                                             self.input_ids.ADJACENCY_IDX.value,
                                             adjacency_idx_size,
                                             acl_resource.run_mode,
                                             device_id,
                                             stream_id,
                                             acl_resource.streams[stream_id])
            check_status("[GCNTask][run] Failed to push adjacency idx to the device.", status)

        status = self.dataset.push_async(mask,
                                         self.input_ids.MASK.value,
                                         mask_size,
                                         acl_resource.run_mode,
                                         device_id,
                                         stream_id,
                                         acl_resource.streams[stream_id])
        check_status("[GCNTask][run] Failed to push mask to the device.", status)

        status = acl.mdl.execute_async(self.dataset.get_model_id(device_id, stream_id),
                                       self.dataset.get_input(device_id, stream_id),
                                       self.dataset.get_output(device_id, stream_id),
                                       acl_resource.streams[stream_id])
        check_status("[GCNTask][run] Failed to run acl.mdl.execute_async.", status)

        status = PostProcessingManager.launch(self.callback,
                                              acl_resource.run_mode,
                                              device_id,
                                              acl_resource.context,
                                              stream_id,
                                              acl_resource.streams[stream_id], [])
        check_status("[GCNTask][run] Failed to submit post processing callback.", status)

        status = acl.rt.synchronize_stream(acl_resource.streams[stream_id])
        check_status("[GCNTask][run] Failed to run acl.rt.synchronize_stream.", status)

        self.post_processing_manager.unsubscribe(device_id, acl_resource.streams)

        return 0

    def callback(self, data):
        """
        Post-process inference result
        """
        run_mode, device_id, _, stream_id, _, _ = data

        output_dims = self.dataset.get_output_dims(device_id, 0)

        gcn_output_data = np.ndarray(output_dims['dims'], dtype=np.float16)
        gcn_output_data_ptr = acl.util.numpy_to_ptr(gcn_output_data)

        status = self.dataset.pull(gcn_output_data_ptr,
                                   0,
                                   self.dataset.get_output_size(device_id, 0),
                                   run_mode,
                                   device_id,
                                   stream_id)
        check_status("[GCNTask][callback] Failed to fetch data from the device.", status)

        predicted_classes = np.argmax(gcn_output_data, axis=1)
        ground_truth = self.labels[self.mask]

        prediction_accuracy = sum(predicted_classes == ground_truth) / len(ground_truth)
        print("[GCN][Inference] Accuracy: " + str(prediction_accuracy))

        GCNTask.__save_outputs(predicted_classes, self.output_path)

    def __get_input_data(self, input_id):
        if self.input_ids.FEATURES_DATA == input_id:
            features_data = self.features[int(InputTypes.DATA.value)]
            features_data_ptr = acl.util.numpy_to_ptr(features_data)
            features_data_size = features_data.size * features_data.itemsize
            return features_data_ptr, features_data_size

        if self.sparse and self.input_ids.FEATURES_IDX == input_id:
            features_idx = self.features[int(InputTypes.IDX.value)]
            features_idx_ptr = acl.util.numpy_to_ptr(features_idx)
            features_idx_size = features_idx.size * features_idx.itemsize
            return features_idx_ptr, features_idx_size

        if self.input_ids.ADJACENCY_DATA == input_id:
            adjacency_data = self.adjacency[int(InputTypes.DATA.value)]
            adjacency_data_ptr = acl.util.numpy_to_ptr(adjacency_data)
            adjacency_data_size = adjacency_data.size * adjacency_data.itemsize
            return adjacency_data_ptr, adjacency_data_size

        if self.sparse and self.input_ids.ADJACENCY_IDX == input_id:
            adjacency_idx = self.adjacency[int(InputTypes.IDX.value)]
            adjacency_idx_ptr = acl.util.numpy_to_ptr(adjacency_idx)
            adjacency_idx_size = adjacency_idx.size * adjacency_idx.itemsize
            return adjacency_idx_ptr, adjacency_idx_size

        if self.input_ids.MASK == input_id:
            mask_ptr = acl.util.numpy_to_ptr(self.mask)
            mask_size = self.mask.size * self.mask.itemsize
            return mask_ptr, mask_size

        raise Exception("Unknown input id " + str(input_id))

    @staticmethod
    def __save_outputs(predicted_classes, output_path):
        predicted_classes_path = output_path + "/predicted_classes.npy"
        np.save(predicted_classes_path, predicted_classes)
        print("[GCN][Inference] Results saved in : " + predicted_classes_path)

    @staticmethod
    def __load_inputs(input_path, sparse):
        features = np.load(input_path + "/features.npy").astype(np.float16)
        labels = np.load(input_path + "/labels.npy")
        adjacency = np.load(input_path + "/adjacency.npy").astype(np.float16)
        mask = np.load(input_path + "/mask.npy")

        features_data, features_idx = GCNTask.__format_input(features, sparse)
        adjacency_data, adjacency_idx = GCNTask.__format_input(adjacency, sparse)

        inference_inputs = \
            [features_data, features_idx], \
            [adjacency_data, adjacency_idx], \
            labels, \
            mask

        return list(inference_inputs)

    @staticmethod
    def __format_input(matrix, sparse):
        if sparse:
            matrix = sp.coo_matrix(matrix, dtype=np.float16)
            data = matrix.data
            idx = np.ascontiguousarray(np.array([matrix.row, matrix.col], dtype=np.int).T)
        else:
            data, idx = matrix, None

        return data, idx

    def get_flops(self):
        """
        Evaluates the flops for the GCN

        (Currently not implemented)
        """
        return
