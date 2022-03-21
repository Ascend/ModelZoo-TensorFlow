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

# Basic imports
import os
import time
import sys
import argparse
import tensorflow as tf
import numpy as np
import multiprocessing as mp

# Tensorflow ops imports
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.training import queue_runner_impl

# Custom imports
from models import *
from utils import initialize_from_dict, save_checkpoint, load_checkpoint, make_dir, Metrics
from utils.logger import Logger
from random import shuffle
from utils.load_dataset_tfrecords import load_dataset
from utils.argument_utils import read_json, assign_args

parser = argparse.ArgumentParser()

# Argument loading

parser.add_argument('--argsFile', action='store', type=str, default='none',
                    help='The name of the file that contains a model\'s arguments. Also requires --model.')

parser.add_argument('--model', action='store', required=True,
                    help='Model architecture (c3d, tsn, i3d, resnet)')

args_init = parser.parse_known_args()[0]
model_name = args_init.model
args_file = args_init.argsFile
args_json = read_json(model_name, args_file)
json_keys = args_json.keys()

# Model parameters

parser.add_argument('--inputDims', action='store', required='inputDims' not in json_keys, type=int,
                    help='Input Dimensions (Number of frames to pass as input to the model)')

parser.add_argument('--outputDims', action='store', required='outputDims' not in json_keys, type=int,
                    help='Output Dimensions (Number of classes in dataset)')

parser.add_argument('--seqLength', action='store', required='seqLength' not in json_keys, type=int,
                    help='Number of output frames expected from model')

parser.add_argument('--modelAlpha', action='store', type=float, default=1.,
                    help='Resampling factor for constant value resampling and alpha initialization')

parser.add_argument('--inputAlpha', action='store', type=float, default=1.,
                    help='Resampling factor for constant value resampling of input video, used mainly for testing models.')

parser.add_argument('--dropoutRate', action='store', type=float, default=0.5,
                    help='Value indicating proability of keeping inputs of the model\'s dropout layers.')

parser.add_argument('--freeze', action='store', type=int, default=0,
                    help='Freeze weights during training of any layers within the model that have the option set. (default False)')

parser.add_argument('--returnLayer', nargs='+', type=str, default=['logits'],
                    help='Which model layers to be returned by the models\' inference during testing.')

parser.add_argument('--loadWeights', action='store', type=str, default='default',
                    help='String which can be used to specify the default weights to load.')

# Experiment parameters

parser.add_argument('--dataset', action='store', required='dataset' not in json_keys,
                    help='Dataset (UCF101, HMDB51)')

parser.add_argument('--loadedDataset', action='store', required='loadedDataset' not in json_keys,
                    help='Dataset (UCF101, HMDB51)')

parser.add_argument('--loadedPreproc', action='store', type=str, default='null',
                    help='The preprocessing of the weights to be loaded.')

parser.add_argument('--numGpus', action='store', type=int, default=1,
                    help='Number of Gpus used for calculation')

parser.add_argument('--gpuList', nargs='+', type=str, default=[],
                    help='List of GPU IDs to be used')

parser.add_argument('--train', action='store', type=int, default=0,
                    help='Binary value to indicate training or evaluation instance')

parser.add_argument('--load', action='store', type=int, default=1,
                    help='Whether to use the current trained checkpoints with the same experiment_name or to test from default weights.')

parser.add_argument('--loadedCheckpoint', action='store', type=int, default=-1,
                    help='Specify the step of the saved model checkpoint that will be loaded for testing. Defaults to most recent checkpoint.')

parser.add_argument('--size', action='store', required='size' not in json_keys, type=int,
                    help='Input frame size')

parser.add_argument('--expName', action='store', required='expName' not in json_keys,
                    help='Unique name of experiment being run')

parser.add_argument('--numVids', action='store', required='numVids' not in json_keys, type=int,
                    help='Number of videos to be used for testing')

parser.add_argument('--split', action='store', type=int, default=1,
                    help='Dataset split to use')

parser.add_argument('--baseDataPath', action='store', default='/z/dat',
                    help='Path to datasets')

parser.add_argument('--fName', action='store', required='fName' not in json_keys,
                    help='Which dataset list to use (trainlist, testlist, vallist)')

parser.add_argument('--clipLength', action='store', type=int, default=-1,
                    help='Length of clips to cut video into, -1 indicates using the entire video as one clip')

parser.add_argument('--videoOffset', action='store', default='none',
                    help='(none or random) indicating where to begin selecting video clips assuming clipOffset is none')

parser.add_argument('--clipOffset', action='store', default='none',
                    help='(none or random) indicating if clips are selected sequentially or randomly')

parser.add_argument('--clipStride', action='store', type=int, default=0,
                    help='Number of frames that overlap between clips, 0 indicates no overlap and negative values indicate a gap of frames between clips')

parser.add_argument('--numClips', action='store', type=int, default=-1,
                    help='Number of clips to break video into, -1 indicates breaking the video into the maximum number of clips based on clipLength, clipStride, and clipOffset')

parser.add_argument('--batchSize', action='store', type=int, default=1,
                    help='Number of clips to load into the model each step.')

parser.add_argument('--metricsDir', action='store', type=str, default='default',
                    help='Name of sub directory within experiment to store metrics. Unique directory names allow for parallel testing.')

parser.add_argument('--metricsMethod', action='store', default='avg_pooling',
                    help='Which method to use to calculate accuracy metrics. (avg_pooling, last_frame, svm, svm_train or extract_features)')

parser.add_argument('--preprocMethod', action='store', default='default',
                    help='Which preprocessing method to use (default, cvr, rr, sr are options for existing models)')

parser.add_argument('--randomInit', action='store', type=int, default=0,
                    help='Randomly initialize model weights, not loading from any files (Default 0)')

parser.add_argument('--avgClips', action='store', type=int, default=0,
                    help='Boolean indicating whether to average predictions across clips (Default 0)')

parser.add_argument('--useSoftmax', action='store', type=int, default=1,
                    help='Boolean indicating whether to apply softmax to the inference of the model (Default 1)')

parser.add_argument('--preprocDebugging', action='store', type=int, default=0,
                    help='Boolean indicating whether to load videos and clips in a queue or to load them directly for debugging (Default 0)')

parser.add_argument('--verbose', action='store', type=int, default=1,
                    help='Boolean switch to display all print statements or not')

parser.add_argument('--topk', action='store', type=int, default=3,
                    help='Integer indication top k predictions made (Default 3)')

parser.add_argument('--save', action='store', type=int, default=1,
                    help='Boolean indicating whether to save any metrics, logs, or results. Used for testing if the code runs.')

parser.add_argument('--reverse', action='store', type=int, default=0,
                    help='Boolean indicating whether reverse videos and classify them as a new action class. 0 all videos are forward, 1 randomly reversed videos, 2 all videos are reversed')

args = parser.parse_args()

args = assign_args(args, args_json, sys.argv)

loaded_preproc = args.loadedPreproc
if loaded_preproc == 'null':
    loaded_preproc = args.preprocMethod

model_name = args.model
save_bool = args.save

model = models_import.create_model_object(modelName=model_name,
                                          inputAlpha=args.inputAlpha,
                                          modelAlpha=args.modelAlpha,
                                          clipLength=args.clipLength,
                                          numVids=args.numVids,
                                          numEpochs=1,
                                          batchSize=args.batchSize,
                                          numClips=args.numClips,
                                          numGpus=1,
                                          train=args.train,
                                          expName=args.expName,
                                          outputDims=args.outputDims,
                                          inputDims=args.inputDims,
                                          preprocMethod=args.preprocMethod,
                                          dropoutRate=args.dropoutRate,
                                          freeze=args.freeze,
                                          loadWeights=args.loadWeights,
                                          verbose=args.verbose)


def test(model, input_dims, output_dims, seq_length, size, dataset,  num_vids, split,
         base_data_path, f_name, return_layer, clip_length, video_offset, clip_offset, num_clips, clip_stride,
         batch_size, verbose, gpu_list, use_softmax, preproc_debugging, reverse):
    """
    Function used to test the performance and analyse a chosen model
    Args:
        :model:              tf-activity-recognition framework model object
        :input_dims:         Number of frames used in input
        :output_dims:        Integer number of classes in current dataset
        :seq_length:         Length of output sequence expected from LSTM
        :size:               List detailing height and width of frame
        :dataset:            Name of dataset being loaded
        :num_vids:           Number of videos to be used for testing
        :split:              Split of dataset being used
        :base_data_path:     Full path to root directory containing datasets
        :f_name:             Specific video directory within a chosen split of a dataset
        :return_layer:       Layer to return from the model, used to extract features
        :clip_length:        Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :video_offset:       String indicating where to begin selecting video clips (provided clipOffset is None)
        :clip_offset:        "none" or "random" indicating where to begin selecting video clips
        :num_clips:          Number of clips to break video into
        :clip_stride:        Number of frames that overlap between clips, 0 indicates no overlap and negative values indicate a gap of frames between clips
        :batch_size:         Number of clips to load into the model each step.
        :loaded_checkpoint:  Specify the exact checkpoint of saved model to be loaded for further training/testing
        :verbose:            Boolean to indicate if all print statement should be procesed or not
        :gpu_list:           List of GPU IDs to be used
        :preproc_method:     The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing
        :use_softmax:        Binary boolean indicating whether to apply softmax to the inference of the model
        :preproc_debugging:  Boolean indicating whether to load videos and clips in a queue or to load them directly for debugging (Default 0)

    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:

        # Initializers for checkpoint and global step variable

        istraining = False
        video_step = tf.Variable(1.0, name='video_step', trainable=istraining)

        # TF session setup
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        # Variables get randomly initialized into tf graph
        sess.run(init)

        data_path = os.path.join(base_data_path, 'tfrecords_' + dataset, 'Split' + str(split), f_name)

        # Setting up tensors for models
        # input_data_tensor - [batchSize, inputDims, height, width, channels]
        input_data_tensor, labels_tensor, names_tensor = load_dataset(model, 1, batch_size, output_dims, input_dims,
                                                                      seq_length, size, data_path, dataset, istraining,
                                                                      clip_length, video_offset, clip_offset, num_clips,
                                                                      clip_stride, video_step, preproc_debugging, 0,
                                                                      verbose, reverse=reverse)

        ######### GPU list check block ####################

        assert (len(gpu_list) <= 1)

        if len(gpu_list) == 0:
            gpu_list = ['0']  # Default choice is ID = 0

        # END IF

        ###################################################

        ################################################## Setup TF graph block ######################################################

        # Model Inference
        with tf.device('/gpu:' + gpu_list[0]):
            logits = model.inference(input_data_tensor[0:batch_size, :, :, :, :],
                                     istraining,
                                     input_dims,
                                     output_dims,
                                     seq_length,
                                     scope,
                                     return_layer=return_layer)[0]

            # Logits shape: [batchSize, seqLength, outputDims] if not, reshape
            logits_shape = logits.get_shape().as_list()
            if (logits_shape[0] != batch_size or logits_shape[1] != seq_length or logits_shape[2] != output_dims) and \
                    return_layer[0] == 'logits':
                logits = tf.reshape(logits, [batch_size, seq_length, output_dims])

            # END IF
            if use_softmax:
                logits = tf.nn.softmax(logits)

            # END IF

        # END WITH

        ############################################################################################################################################

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

        # Variables get randomly initialized into tf graph
        sess.run(init)

        count = 0
        videos_loaded = 0

        print("Begin Testing")

        # END IF

        ########################################## Testing loop block ################################################################

        while videos_loaded <= num_vids:
            output_predictions, labels, names,x = sess.run([logits, labels_tensor, names_tensor, input_data_tensor[0:batch_size,:,:,:,:]])
            num_input_bin = str(videos_loaded).zfill(len(str(num_vids)))
            x.tofile(base_data_path+"/input_bin/{}_input.bin".format(num_input_bin))

            videos_loaded += 1

                # Extract remaining clips from currently loaded video, once it finishes exit while loop
            if videos_loaded > num_vids:
                break

            count += 1
        print("DONE")

        # END WHILE

        #########################################################################################################################################################

    # END WITH

    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
    if not args.train:
        test(model=model,
             input_dims=args.inputDims,
             output_dims=args.outputDims,
             seq_length=args.seqLength,
             size=[args.size, args.size],
             dataset=args.dataset,
             num_vids=args.numVids,
             split=args.split,
             base_data_path=args.baseDataPath,
             f_name=args.fName,
             return_layer=args.returnLayer,
             clip_length=args.clipLength,
             video_offset=args.videoOffset,
             clip_offset=args.clipOffset,
             num_clips=args.numClips,
             clip_stride=args.clipStride,
             batch_size=args.batchSize,
             verbose=args.verbose,
             gpu_list=args.gpuList,

             use_softmax=args.useSoftmax,
             preproc_debugging=args.preprocDebugging,
             reverse=args.reverse)

    # END IF

