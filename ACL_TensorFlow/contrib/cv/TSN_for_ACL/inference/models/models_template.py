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

" MODELNAME MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import tensorflow as tf
import numpy      as np

# Try imports assuming regular framework structure
try:
    from models.models_abstract      import Abstract_Model_Class
    from utils.layers_utils          import *

# Try imports assuming call is coming from models/current_model/current_model.py
except:
    import sys
    sys.path.append('../../')
    from models.models_abstract import Abstract_Model_Class
    from utils.layers_utils     import *

# END TRY

from models.tsn.default_preprocessing      import preprocess


class MODELNAME(Abstract_Model_Class):

    def __init__(self, **kwargs):
        """
        Args:
            Pass all arguments on to parent class, you may not add additional arguments without modifying abstract_model_class.py, Models.py, train.py, and inference.py. Enter any additional initialization functionality here if desired.
        """
        super(MODELNAME, self).__init__(**kwargs)


    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, batch_size, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [BatchSize x Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: String matching name of a layer in current model
            :weight_decay: Double value of weight decay
            :batch_size:   Number of videos or clips to process at a time

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                       Add MODELNAME Network Layers HERE                  #
        ############################################################################

        if self.verbose:
            print('Generating MODELNAME network layers')

        # END IF

        with tf.name_scope(scope, 'MODELNAME', [inputs]):
            layers = {}

            ########################################################################################
            #        TODO: Add any desired layers from layers_utils to this layers dictionary      #
            #                                                                                      #
            #       EX: layers['conv1'] = conv3d_layer(input_tensor=inputs,                        #
            #           filter_dims=[dim1, dim2, dim3, dim4],                                      #
            #           name=NAME,                                                                 #
            #           weight_decay = wd)                                                         #
            ########################################################################################


            ########################################################################################
            #       TODO: Final Layer must be 'logits'                                             #
            #                                                                                      #
            #  EX:  layers['logits'] = [fully_connected_layer(input_tensor=layers['previous'],     #
            #                                         out_dim=output_dims, non_linear_fn=None,     #
            #                                         name='out', weight_decay=weight_decay)]      #
            ########################################################################################

            layers['logits'] = # TODO Every model must return a layer named 'logits'

            layers['logits'] = tf.reshape(layers['logits'], [batch_size, seq_length, output_dims])

        # END WITH

        return [layers[x] for x in return_layer]




    # def load_default_weights(self):
    #     """
    #     return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
    #     """
    #
    #     ############################################################################
    #     # TODO: Add default model weights to models/weights/ and import them here  #
    #     #                          ( OPTIONAL )                                    #
    #     #                                                                          #
    #     # EX: return np.load('models/weights/model_weights.npy')                   #
    #     #                                                                          #
    #     ############################################################################




    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step):
        """
        Args:
            :input_data_tensor:     Data loaded from tfrecords containing either video or clips
            :frames:                Number of frames in loaded video or clip
            :height:                Pixel height of loaded video or clip
            :width:                 Pixel width of loaded video or clip
            :channel:               Number of channels in video or clip, usually 3 (RGB)
            :input_dims:            Number of frames used in input
            :output_dims:           Integer number of classes in current dataset
            :seq_length:            Length of output sequence
            :size:                  List detailing values of height and width for final frames
            :label:                 Label for loaded data
            :is_training:           Boolean value indication phase (TRAIN OR TEST)
            :video_step:            Tensorflow variable indicating the total number of videos (not clips) that have been loaded
        """

        ####################################################
        # TODO: Add more preprcessing arguments if desired #
        ####################################################

        return preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step, self.input_alpha)



    """ Function to return loss calculated on given network """
    def loss(self, logits, labels, loss_type):
        """
        Args:
           :logits:     Unscaled logits returned from final layer in model
           :labels:     True labels corresponding to loaded data
           :loss_type:  Allow for multiple losses that can be selected at run time. Implemented through if statements
        """

        ####################################################################################
        #  TODO: ADD CUSTOM LOSS HERE, DEFAULT IS CROSS ENTROPY LOSS                       #
        #                                                                                  #
        #   EX: labels = tf.cast(labels, tf.int64)                                         #
        #       cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, #
        #                                                            logits=logits)        #
        #        return cross_entropy_loss                                                 #
        ####################################################################################
