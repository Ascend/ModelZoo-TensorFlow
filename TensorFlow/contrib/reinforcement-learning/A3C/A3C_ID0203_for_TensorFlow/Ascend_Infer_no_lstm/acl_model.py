"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-28 14:04:45
"""
import acl
import struct
import numpy as np
from constants import *
from utils import *

class Model(object):
    def __init__(self, run_mode, model_path):
        self._run_mode = run_mode
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.input_dataset = None
        self.output_dataset = None
        self._output_info = []
        self.model_desc = None          # pointer when using
        self.init_resource()
        self.data = None

    def __del__(self):
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)
        print("Model release source success")

    def init_resource(self):
        print("[Model] class Model init resource stage:")
        print ("-----------------------------------------")
        print ("self.model_path = ",self.model_path)
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_output_dataset(output_size)
        print("[Model] class Model init resource stage success")
        self._get_output_info(output_size)

        return SUCCESS

    def _get_output_info(self, output_size):
        for i in range(output_size):
            dims = acl.mdl.get_output_dims(self.model_desc, i)
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)
            self._output_info.append({"shape": tuple(dims[0]["dims"]), "type": datatype})
 
    def _gen_output_dataset(self, size):
        print("[Model] create model output dataset:")
        dataset = acl.mdl.create_dataset()
        for i in range(size):
            temp_buffer_size = acl.mdl.\
                get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            #print(temp_buffer_size)
            dataset_buffer = acl.create_data_buffer(temp_buffer,
                                                    temp_buffer_size)              
            _, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
            if ret:
                acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)
        self.output_dataset = dataset
        print("[Model] create model output dataset success")

    def _gen_input_dataset(self, data, data_size):
        self.input_dataset = acl.mdl.create_dataset()
        input_dataset_buffer = acl.create_data_buffer(data, data_size)
        _, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset,
            input_dataset_buffer)
        
        if ret:
            ret = acl.destroy_data_buffer(self.input_dataset)
            check_ret("acl.destroy_data_buffer", ret)

    def _gen_input_dataset_from_array(self, s):
        self.input_dataset = acl.mdl.create_dataset()
        #print(s.shape)
        ptr = acl.util.numpy_to_ptr(s)
        size = s.size * s.itemsize
        #print(self._output_info)
        if self.data == None:
            self.data = copy_data_device_to_device(ptr, size)
        else:
            ret = acl.rt.memcpy(self.data, size, ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE)
            if ret != ACL_ERROR_NONE:
                print("Copy model %dth input to device failed"%(index))

        if self.data is None:
            print("Malloc memory and copy model input to device failed"%(index))

        dataset_buffer = acl.create_data_buffer(self.data, size)
        _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, dataset_buffer)
        
        if ret:
            ret = acl.destroy_data_buffer(self.input_dataset)
            check_ret("acl.destroy_data_buffer", ret)

    def execute(self, data):
        self._gen_input_dataset_from_array(data)
        #print(self._output_info)
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_dataset)
        check_ret("acl.mdl.execute", ret)
        return self._output_dataset_to_numpy()

    def _output_dataset_to_numpy(self):
        dataset = []
        num = acl.mdl.get_dataset_num_buffers(self.output_dataset)
        for i in range(num):
            buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data = acl.get_data_buffer_addr(buffer)
            size = acl.get_data_buffer_size(buffer)
            if self._run_mode == ACL_HOST:
                data = copy_data_device_to_host(data, size)
            
            data_array = acl.util.ptr_to_numpy(data, (size, ), NPY_BYTE)
            data_array = self._unpack_output_item(data_array,
                                                  self._output_info[i]["shape"],
                                                  self._output_info[i]["type"])
            dataset.append(data_array)
        return dataset  

    def _unpack_output_item(self, byte_array, shape, datatype):
        tag = ""
        np_type = None
        if datatype == ACL_FLOAT:
            tag = 'f'
            np_type = np.float
        elif datatype == ACL_INT32:
            tag = 'I'
            np_type = np.int32
        elif datatype == ACL_UINT32:
            tag = 'I'
            np_type = np.uint32
        else:
            print("unsurpport datatype ", datatype);
            return
        size = byte_array.size/4
        unpack_tag = str(int(size)) + tag
        #print(datatype==ACL_INT32)
        #print(np_type)
        st = struct.unpack(unpack_tag,bytearray(byte_array))
        return np.array(st).astype(np_type).reshape(shape)
        #return byte_array[0]

    def _release_dataset(self):
        for dataset in [self.input_dataset, self.output_dataset]:
            if not dataset:
                continue
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)


