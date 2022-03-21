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


import os

import numpy      as np
import tensorflow as tf

def load_checkpoint(model, dataset, experiment_name, loaded_checkpoint, preproc_method):
    """
    Function to checkpoint file (both ckpt text file, numpy and dat file)
    Args:
        :model:                 String indicating selected model
        :dataset:               String indicating selected dataset
        :experiment_name:       Name of experiment folder
        :loaded_checkpoint:     Number of the checkpoint to be loaded, -1 loads the most recent checkpoint
        :preproc_method:     The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing

    Return:
        numpy containing model parameters, global step and learning rate saved values.
    """
    if loaded_checkpoint == -1:
        try:
            checkpoints_file = os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints', 'checkpoint')
            f = open(checkpoints_file, 'r')
            filename = f.readline()
            f.close()
            filename = filename.split(' ')[1].split('\"')[1]

        except:
            print("Failed to load checkpoint information file")
            exit()

        # END TRY

    else:
        filename = 'checkpoint-'+str(loaded_checkpoint)

    try:
        gs_init = int(filename.split('-')[1])
        ckpt = np.load(os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints',filename+'.npy'))

    except:
        print("Failed to load saved checkpoint numpy file: ", filename)
        exit()

    # END TRY

    try:
        data_file = open(os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints', filename+'.dat'), 'r')
        data_str = data_file.readlines()
        for data in data_str:
            if ':' in data:
                data_name, data_value = data.split(':')

            else:
                data_name = data[:2]
                data_value = data[3:]

            if data_name == "lr":
                lr_init = float(data_value)

            # END IF

        # END FOR

        data_file.close()

        return ckpt, gs_init, lr_init

    except:
        print("Failed to extract checkpoint data: ", filename)
        exit()

    # END TRY

def save_checkpoint(sess, model, dataset, experiment_name, preproc_method, lr, gs):
    """
    Function to save numpy checkpoint file
    Args:
        :sess:            Tensorflow session instance
        :model:           String indicating selected model
        :dataset:         String indicating selected dataset
        :experiment_name: Name of experiment folder
        :preproc_method:  The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing
        :lr:              Learning rate
        :gs:              Current global step

    Return:
       Does not return anything
    """

    filename = 'checkpoint-'+str(gs)

    c_file = open(os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints','checkpoint'), 'w')
    c_file.write('model_checkpoint_path: "'+filename+'"')
    c_file.close()

    data_file = open(os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints', filename+'.dat'), 'w')
    data_file.write('lr:'+str(lr)+'\n')
    data_file.close()

    data_dict = {}

    for var in tf.global_variables():
        layers = var.name.split('/')
        data_dict = _add_tensor(data_dict, layers, sess.run(var))

    # END FOR

    np.save(os.path.join('results', model, dataset, preproc_method, experiment_name, 'checkpoints',filename+'.npy'), data_dict)


def _add_tensor(data_dict, keys_list, data):
    """
    Function recursively builds the dictionary with a layered structure
    Args:
        :data_dict: Data dictionary to be updated
        :keys_list: List of string indicating depth of dictionary to be built and corresponding items
        :data:      String indicating selected dataset

    Return:
        Returns a dictionary containing model parameters
    """

    if len(keys_list) == 0:
        return data

    else:
        try:
            curr_data_dict = data_dict[keys_list[0]]
        except:
            curr_data_dict = {}

            # END TRY

        data_dict[keys_list[0]] = _add_tensor(curr_data_dict, keys_list[1:], data)
        return data_dict

        # END IF


def _assign_tensors(sess, curr_dict, tensor_name):
    """
    Function recursively assigns model parameters their values from a given dictionary
    Args:
        :sess:        Tensorflow session instance
        :curr_dict:   Dictionary containing model parameter values
        :tensor_name: String indicating name of tensor to be assigned values

    Return:
       Does not return anything
    """
    try:
        if type(curr_dict) == type({}):
            for key in curr_dict.keys():
                _assign_tensors(sess, curr_dict[key], tensor_name+'/'+key)

            # END FOR

        else:
            if ':' not in tensor_name:
                tensor_name = tensor_name + ':0'

            # END IF

            if 'weights' in tensor_name:
                tensor_name = tensor_name.replace('weights', 'kernel')

            # END IF

            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(tensor_name), curr_dict))

        # END IF

    except:
        if 'Momentum' not in tensor_name:
            print("Notice: Tensor " + tensor_name + " could not be assigned properly. The tensors' default initializer will be used if possible. Verify the shape and name of the tensor.")

        #END IF

    # END TRY


def initialize_from_dict(sess, data_dict, model_name):
    """
    Function initializes model parameters from value given in a dictionary
    Args:
        :sess:        Tensorflow session instance
        :data_dict:   Dictionary containing model parameter values

    Return:
       Does not return anything
    """

    print('Initializing model weights...')
    try:
        data_dict = data_dict.tolist()
        for key in data_dict.keys():
            _assign_tensors(sess, data_dict[key], key)

        # END FOR

    except:
        print("Error: Failed to initialize saved weights. Ensure naming convention in saved weights matches the defined model.")
        exit()

    # END TRY
