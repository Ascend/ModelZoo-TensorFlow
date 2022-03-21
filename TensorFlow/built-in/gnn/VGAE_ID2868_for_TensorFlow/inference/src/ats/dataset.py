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
ATS Framework

Container for input and output datasets
"""

import numpy as np

import acl

from ats.constants import MAX_DEVICE_COUNT, ATS_SEND, ATS_RECV
from ats.utils import check_status

from ats.constants import ACL_HOST
from ats.constants import ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_HOST_TO_DEVICE
from ats.constants import ACL_MEMCPY_DEVICE_TO_DEVICE
from ats.constants import ACL_MEM_MALLOC_NORMAL_ONLY


class DataSet(object):
    # pylint: disable=too-many-instance-attributes
    """
    Implements a container to be used for input and output datasets.
    The datasets are dynamically discovered by querying the model.
    """

    def __init__(self, model_path, name=""):
        self.name = name
        self.model_path = model_path
        self.is_dataset_created = [False]*MAX_DEVICE_COUNT

        self.model_ids = [np.ndarray(shape=(1), dtype=int)]*MAX_DEVICE_COUNT

        self.model_descs = [np.ndarray(shape=(1), dtype=int)]*MAX_DEVICE_COUNT

        self.input_buffers = [np.ndarray(shape=(1, 1), dtype=int)]*MAX_DEVICE_COUNT
        self.output_buffers = [np.ndarray(shape=(1, 1), dtype=int)]*MAX_DEVICE_COUNT

        self.input_datasets = [np.ndarray(shape=(1), dtype=int)]*MAX_DEVICE_COUNT
        self.output_datasets = [np.ndarray(shape=(1), dtype=int)]*MAX_DEVICE_COUNT

        self.input_dims = [[]]*MAX_DEVICE_COUNT
        self.output_dims = [[]]*MAX_DEVICE_COUNT

        self.input_sizes = [0]*MAX_DEVICE_COUNT
        self.output_sizes = [0]*MAX_DEVICE_COUNT

    def push(self, data, input_id, size, run_mode, device_id, stream_id):
        """
        Synchronous push of data to a device
        """
        return acl.rt.memcpy(int(self.input_buffers[device_id][stream_id][input_id]),
                             size,
                             data,
                             size,
                             DataSet.__get_memcpy_kind(ATS_SEND, run_mode))

    def push_async(self, data, input_id, size, run_mode, device_id, stream_id, stream):
        """
        Asynchronous push of data to a device
        """
        return acl.rt.memcpy_async(int(self.input_buffers[device_id][stream_id][input_id]),
                                   size,
                                   data,
                                   size,
                                   DataSet.__get_memcpy_kind(ATS_SEND, run_mode),
                                   stream)

    def pull(self, data, output_id, size, run_mode, device_id, stream_id):
        """
        Synchronous pull of data from a device
        """
        return acl.rt.memcpy(data,
                             size,
                             int(self.output_buffers[device_id][stream_id][output_id]),
                             size,
                             self.__get_memcpy_kind(ATS_RECV, run_mode))

    def pull_async(self, data, output_id, size, run_mode, device_id, stream_id, stream):
        """
        Asynchronous pull of data from a device
        """
        return acl.rt.memcpy_async(data,
                                   size,
                                   int(self.output_buffers[device_id][stream_id][output_id]),
                                   size,
                                   self.__get_memcpy_kind(ATS_RECV, run_mode),
                                   stream)

    def create(self, device_id, stream_count):
        """
        Create the datasets:
            - load model
            - create input dataset
            - create output dataset
        """
        if self.is_dataset_created[device_id]:
            return 0

        self.__load(device_id, stream_count)
        self.__create_input(device_id, stream_count)
        self.__create_output(device_id, stream_count)

        return 0

    def destroy(self, device_id, stream_count):
        """
        Destroy the datasets:
            - destroy input dataset
            - destroy output dataset
            - unload model
        """
        if self.is_dataset_created[device_id]:
            return 0

        self.__destroy_input(device_id, stream_count)
        self.__destroy_output(device_id, stream_count)
        self.__unload(device_id, stream_count)

        return 0

    def get_model_id(self, device_id, stream_id):
        """
        Return the model ID on the specified device and stream
        """
        return self.model_ids[device_id][stream_id]

    def get_input(self, device_id, stream_id):
        """
        Return the handle to the input dataset
        """
        return int(self.input_datasets[device_id][stream_id])

    def get_output(self, device_id, stream_id):
        """
        Return the handle to the output dataset
        """
        return int(self.output_datasets[device_id][stream_id])

    def get_output_size(self, device_id, output_id):
        """
        Return the size (in bytes) of the output dataset
        """
        return acl.mdl.get_output_size_by_index(self.model_descs[device_id], output_id)

    def get_input_dims(self, device_id, input_id):
        """
        Return the dimensions of the input tensor
        """
        return self.input_dims[device_id][input_id]

    def get_output_dims(self, device_id, output_id):
        """
        Return the dimensions of the output tensor
        """
        return self.output_dims[device_id][output_id]

    @staticmethod
    def __get_memcpy_kind(direction, run_mode):
        if ACL_HOST == run_mode:
            if ATS_RECV == direction:
                return ACL_MEMCPY_DEVICE_TO_HOST
            return ACL_MEMCPY_HOST_TO_DEVICE
        return ACL_MEMCPY_DEVICE_TO_DEVICE

    def __load(self, device_id, stream_count):
        self.model_ids[device_id] = np.resize(self.model_ids[device_id], (stream_count))
        for stream_id in range(stream_count):
            self.model_ids[device_id][stream_id], status = acl.mdl.load_from_file(self.model_path)
            check_status("[DataSet][__load] Failed to load model.", status)

        self.model_descs[device_id] = acl.mdl.create_desc()
        status = acl.mdl.get_desc(self.model_descs[device_id], self.model_ids[device_id][0])

        self.input_sizes[device_id] = acl.mdl.get_num_inputs(self.model_descs[device_id])
        self.output_sizes[device_id] = acl.mdl.get_num_outputs(self.model_descs[device_id])

        self.input_dims[device_id] = [None]*self.input_sizes[device_id]
        for input_id in range(self.input_sizes[device_id]):
            input_dims, status = acl.mdl.get_input_dims(self.model_descs[device_id], input_id)
            self.input_dims[device_id][input_id] =  input_dims
            check_status("[DataSet][__load] Failed to get input dims.", status)

        self.output_dims[device_id] = [None]*self.output_sizes[device_id]
        for output_id in range(self.output_sizes[device_id]):
            output_dims, status = acl.mdl.get_output_dims(self.model_descs[device_id], output_id)
            self.output_dims[device_id][output_id] = output_dims
            check_status("[DataSet][__load] Failed to get output dims.", status)

        self.input_datasets[device_id] = np.resize(self.input_datasets[device_id],
                                                   (stream_count))
        self.output_datasets[device_id] = np.resize(self.output_datasets[device_id],
                                                    (stream_count))

        self.input_buffers[device_id] = np.resize(self.input_buffers[device_id],
                                                  (stream_count, self.input_sizes[device_id]))
        self.output_buffers[device_id] = np.resize(self.output_buffers[device_id],
                                                   (stream_count, self.output_sizes[device_id]))

        return 0

    def __unload(self, device_id, stream_count):
        acl.mdl.destroy_desc(self.model_descs[device_id])
        for stream_id in range(stream_count):
            acl.mdl.unload(self.model_ids[device_id][stream_id])

        return 0

    def __create_input(self, device_id, stream_count):
        for stream_id in range(stream_count):
            self.input_datasets[device_id][stream_id] = acl.mdl.create_dataset()
            for input_id in range(self.input_sizes[device_id]):
                input_buffer_size = acl.mdl.get_input_size_by_index(self.model_descs[device_id],
                                                                    input_id)
                input_buffer, status = acl.rt.malloc(input_buffer_size,
                                                     ACL_MEM_MALLOC_NORMAL_ONLY)
                self.input_buffers[device_id][stream_id][input_id] = input_buffer
                check_status("[DataSet][__create_input] Failed to allocate input buffer.", status)

                input_data = acl.create_data_buffer(int(input_buffer), input_buffer_size)
                acl.mdl.add_dataset_buffer(int(self.input_datasets[device_id][stream_id]),
                                           input_data)

        return 0

    def __create_output(self, device_id, stream_count):
        for stream_id in range(stream_count):
            self.output_datasets[device_id][stream_id] = acl.mdl.create_dataset()
            for output_id in range(self.output_sizes[device_id]):
                output_buffer_size = acl.mdl.get_output_size_by_index(self.model_descs[device_id],
                                                                      output_id)
                output_buffer, status = acl.rt.malloc(output_buffer_size,
                                                      ACL_MEM_MALLOC_NORMAL_ONLY)
                self.output_buffers[device_id][stream_id][output_id] = output_buffer
                check_status("[DataSet][__create_output] Failed to allocate output buffer.", status)

                output_data = acl.create_data_buffer(int(output_buffer), output_buffer_size)
                acl.mdl.add_dataset_buffer(int(self.output_datasets[device_id][stream_id]),
                                           output_data)

        return 0

    def __destroy_input(self, device_id, stream_count):
        for stream_id in range(stream_count):
            for input_id in range(self.input_sizes[device_id]):
                input_dataset = self.input_datasets[device_id][stream_id]
                input_data = acl.mdl.get_dataset_buffer(int(input_dataset), input_id)
                input_buffer = acl.get_data_buffer_addr(input_data)
                acl.destroy_data_buffer(input_data)
                acl.rt.free(input_buffer)

    def __destroy_output(self, device_id, stream_count):
        for stream_id in range(stream_count):
            for output_id in range(self.output_sizes[device_id]):
                output_dataset = self.output_datasets[device_id][stream_id]
                output_data = acl.mdl.get_dataset_buffer(int(output_dataset), output_id)
                output_buffer = acl.get_data_buffer_addr(output_data)
                acl.destroy_data_buffer(output_data)
                acl.rt.free(output_buffer)
