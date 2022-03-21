# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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


from npu_bridge.npu_init import *
import pickle
import argparse
from deltaencoder import DeltaEncoder


if __name__ == "__main__":
    ########### Parse Argument ################
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--path", type=str, default='./data/mIN.pkl')
    parser.add_argument("--data_set", type=str, default='mIN')
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_ways", type=int, default=5)
    # parser.add_argument("--verbose", type=bool, default=True)
    pars_args = parser.parse_args()
    # print(pars_args.num_shots)
    # print(pars_args.num_epoch)
    # print(pars_args.batch_size)
    # print(pars_args.num_ways)


    ########### Load Data ################
    features_train, labels_train, features_test, labels_test, episodes_1shot, episodes_5shot = pickle.load(open(pars_args.path,'rb'),encoding='iso-8859-1')

    # features_train/features_test are features extracted from some backbone (resnet18); they are np array with size = (N,D), where N is the number of samples and D the features dimensions
    # labels_train/labels_test are one hot GT labels with size = (N,C), where C is the number of classes (can be different for train and test sets
    # episodes_*shot are supplied for reproduction of the paper results size=(num_episodes, num_classes, num_shots, D)


    ######### 1-shot Experiment #########
    args = {'data_set' : 'mIN',
            'num_shots' : 1,
            'num_epoch': 6,
            'nb_val_loop': 10,
            'learning_rate': 1e-5, 
            'drop_out_rate': 0.5,
            'drop_out_rate_input': 0.0,
            'batch_size': 128,
            'noise_size' : 16,
            'nb_img' : 1024,
            'num_ways' : 5,
            'encoder_size' : [8192],
            'decoder_size' : [8192],
            'opt_type': 'adam'
           }
    
    args['data_set'] = pars_args.data_set
    args['num_shots'] = pars_args.num_shots
    args['num_epoch'] = pars_args.num_epoch
    args['batch_size'] = pars_args.batch_size
    args['num_ways'] = pars_args.num_ways

    # print(args)

    if pars_args.num_shots==1:
        model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_1shot)
    else:
        model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_5shot)

    model.train(verbose=True)


