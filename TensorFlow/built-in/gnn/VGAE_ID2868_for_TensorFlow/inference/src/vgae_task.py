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
# Author: Salli Moustafa 
"""
Inference logic for the VGAE
"""

import os
from enum import Enum
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import acl

from ats.abstract_task import AbstractTask
from ats.post_processing_manager import PostProcessingManager
from ats.utils import check_status


class VGAEInputTypes(Enum):
    """
    Input types
    """
    INDICES = 0
    VALUES = 1
    SHAPE = 2


class VGAEInputIds(Enum):
    """
    VGAE Input IDs
    """
    FEATURES_INDICES = 0
    FEATURES_VALUES = 1
    FEATURES_SHAPE = 2
    ADJACENCY_INDICES = 3
    ADJACENCY_VALUES = 4
    ADJACENCY_SHAPE = 5
    EDGES_POS = 6
    EDGES_NEG = 7


class VGAEOutputIds(Enum):
    """
    VGAE Output IDs
    """
    PREDICTIONS = 0
    LABELS = 1


class VGAETask(AbstractTask):
    """
    Implements the core logic for the VGAE inference as expected by ATS
    """
    def __init__(self, dataset, input_path, output_path):
        self.dataset = dataset
        self.input_path = input_path
        self.output_path = output_path

        self.features, self.adjacency, self.edges_pos, self.edges_neg \
            = \
            VGAETask.__load_inputs(os.path.join(input_path, "inputs_" + self.dataset.name + ".npy"))

        self.post_processing_manager = PostProcessingManager(1, 100)

    def init(self, device_id, acl_resource):
        """
        Instantiates the model on the current device
        """

        self.post_processing_manager.init(device_id)

        self.dataset.create(device_id, acl_resource.stream_count)

    def finalize(self, device_id, stream_count):
        """
        Finalizes the model and free acquired resources
        """

        self.dataset.destroy(device_id, stream_count)

        self.post_processing_manager.finalize(device_id)

    def run(self, device_id, acl_resource):
        """
        VGAE inference workflow
        """

        stream_id = 0

        self.post_processing_manager.subscribe(device_id,
                                               acl_resource.context,
                                               acl_resource.streams)

        self.__push_data(acl_resource.run_mode, device_id, stream_id, acl_resource.streams)

        status = acl.mdl.execute_async(self.dataset.get_model_id(device_id, stream_id),
                                       self.dataset.get_input(device_id, stream_id),
                                       self.dataset.get_output(device_id, stream_id),
                                       acl_resource.streams[stream_id])
        check_status("[VGAETask][run] Failed to run acl.mdl.execute_async.", status)

        status = PostProcessingManager.launch(self.callback,
                                              acl_resource.run_mode,
                                              device_id,
                                              acl_resource.context,
                                              stream_id,
                                              acl_resource.streams[stream_id], [])
        check_status("[VGAETask][run] Failed to submit post processing callback.", status)

        status = acl.rt.synchronize_stream(acl_resource.streams[stream_id])
        check_status("[VGAETask][run] Failed to run acl.rt.synchronize_stream.", status)

        self.post_processing_manager.unsubscribe(device_id, acl_resource.streams)

        return 0

    def callback(self, data):
        """
        Post-process inference result
        """
        run_mode, device_id, _, stream_id, _, _ = data

        vgae_outputs = self.__pull_data(run_mode, device_id, stream_id)

        roc_score = roc_auc_score(vgae_outputs[VGAEOutputIds.LABELS.name],
                                  vgae_outputs[VGAEOutputIds.PREDICTIONS.name])
        ap_score = average_precision_score(vgae_outputs[VGAEOutputIds.LABELS.name],
                                           vgae_outputs[VGAEOutputIds.PREDICTIONS.name])

        print("[" + self.dataset.name + "] ROC score: {}".format(roc_score))
        print("[" + self.dataset.name + "] AP score: {}".format(ap_score))

        VGAETask.__save_outputs(vgae_outputs, self.output_path)

    def __push_data(self, run_mode, device_id, stream_id, streams):
        for input_id in VGAEInputIds:
            [input_data_ptr, input_data_size] = self.__get_input_data(input_id)
            status = self.dataset.push_async(input_data_ptr,
                                             input_id.value,
                                             input_data_size,
                                             run_mode,
                                             device_id,
                                             stream_id,
                                             streams[stream_id])
            check_status("[VGAETask][run] Failed to push " + input_id.name + "to the device.", status)
        status = acl.rt.synchronize_stream(streams[stream_id])
        check_status("[VGAETask][run] Failed to run acl.rt.synchronize_stream.", status)

    def __pull_data(self, run_mode, device_id, stream_id):
        outputs = {}
        
        for output_id in VGAEOutputIds:
            output_dims = self.dataset.get_output_dims(device_id, output_id.value)
            output_data = np.ndarray(output_dims['dims'], dtype=np.float16)
            output_data_ptr = acl.util.numpy_to_ptr(output_data)

            status = self.dataset.pull(output_data_ptr,
                                       output_id.value,
                                       self.dataset.get_output_size(device_id, output_id.value),
                                       run_mode,
                                       device_id,
                                       stream_id)
            check_status("[VGAETask][callback] Failed to fetch" + output_id.name + "from the device.", status)

            outputs[output_id.name] = output_data

        return outputs

    def __get_data_ptr(self, input):
        input_ptr = acl.util.numpy_to_ptr(input)
        input_size = input.size * input.itemsize
        return [input_ptr, input_size]

    def __get_input_data(self, input_id):
        if VGAEInputIds.FEATURES_INDICES == input_id:
            return self.__get_data_ptr(self.features[VGAEInputTypes.INDICES.value])

        if VGAEInputIds.FEATURES_VALUES == input_id:
            return self.__get_data_ptr(self.features[VGAEInputTypes.VALUES.value])

        if VGAEInputIds.FEATURES_SHAPE == input_id:
            return self.__get_data_ptr(self.features[VGAEInputTypes.SHAPE.value])

        if VGAEInputIds.ADJACENCY_INDICES == input_id:
            return self.__get_data_ptr(self.adjacency[VGAEInputTypes.INDICES.value])

        if VGAEInputIds.ADJACENCY_VALUES == input_id:
            return self.__get_data_ptr(self.adjacency[VGAEInputTypes.VALUES.value])

        if VGAEInputIds.ADJACENCY_SHAPE == input_id:
            return self.__get_data_ptr(self.adjacency[VGAEInputTypes.SHAPE.value])

        if VGAEInputIds.EDGES_POS == input_id:
            return self.__get_data_ptr(self.edges_pos)

        if VGAEInputIds.EDGES_NEG == input_id:
            return self.__get_data_ptr(self.edges_neg)

    @staticmethod
    def __save_outputs(vgae_outputs, vgae_output_path):
        for output_id in VGAEOutputIds:
            output_name = output_id.name
            output_path = vgae_output_path + "/vgae_" + output_name.lower() + ".npy"
            np.save(output_path, vgae_outputs[output_name])
            print("[VGAE][Inference] " + output_name + " saved in : " + output_path)

    @staticmethod
    def __load_inputs(input_path):
        input_files = np.load(input_path, allow_pickle=True).item()

        features = input_files['features']
        adjacency = input_files['adj_norm']
        edges_pos = input_files['val_edges']
        edges_neg = input_files['val_edges_false']

        features_indices, features_values, features_shape = VGAETask.__format_input(features)
        adjacency_indices, adjacency_values, adjacency_shape = VGAETask.__format_input(adjacency)

        inference_inputs = \
            [features_indices, features_values, features_shape], \
            [adjacency_indices, adjacency_values, adjacency_shape], \
            edges_pos, \
            edges_neg

        return list(inference_inputs)

    @staticmethod
    def __format_input(matrix):
        indices = np.ascontiguousarray(np.array(matrix[VGAEInputTypes.INDICES.value], dtype=np.int))
        values = matrix[VGAEInputTypes.VALUES.value].astype(np.float16)
        shape = np.ascontiguousarray(np.array(matrix[VGAEInputTypes.SHAPE.value], dtype=np.int))

        return indices, values, shape

    def get_flops(self):
        """
        Evaluates the flops for the VGAE

        (Currently not implemented)
        """
        return
