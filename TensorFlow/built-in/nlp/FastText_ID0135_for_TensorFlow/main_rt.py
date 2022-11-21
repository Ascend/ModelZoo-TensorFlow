#
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
#
import json
import numpy as np
from sklearn.model_selection import train_test_split
from word_embeddings_rt import load_data,prepare_data_for_word_vectors,building_word_vector_model,\
classification_model,padding_input,prepare_data_for_word_vectors_imdb,ELMoEmbedding,data_prep_ELMo,Classification_model_with_ELMo
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', default='/root/.keras/datasets/', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate for adam')
parser.add_argument('--loss_scale', dest='loss_scale', default='True', help='enable loss scale ,default is True')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
args = parser.parse_args()
def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set

with open("config.json","r") as f:
    params_set = json.load(f)
params_set = json_to_dict(params_set)


with open("model_params.json", "r") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)

'''
    load_data function works on imdb data. In order to load your data, comment line 27 and pass your data in the form of X,y
    X = text data column
    y = label column(0,1 etc)

'''
# for imdb data
if params_set["option"]in [0,1,2]:
    x_train,x_test,y_train,y_test = load_data(args.data_path, params_set["vocab_size"],params_set["max_len"])
    sentences,word_ix = prepare_data_for_word_vectors_imdb(args.data_path, x_train)
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y_train)

    # for other data:
    # put your data in the form of X,y
    '''
    X = ["this is a sentence","this is another sentence by me","yet another sentence for training","one more again"]
    y=np.array([0,1,1,0])

    sentences_as_words,sentences,word_ix = prepare_data_for_word_vectors(X)
    print("sentences loaded")
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)


    print("word vector model built")
    x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=params_set["split_ratio"], random_state=42)
    print("Data split done")
    '''
    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])

    model = classification_model(args,params_set["embed_dim"],x_train_pad,x_test_pad,y_train,y_test,
                                 params_set["vocab_size"],word_ix,model_wv,
                                 params_set["trainable_param"],
                                 params_set["option"])
    print(model.summary())

else:
    x_train,x_test,y_train,y_test = load_data(args.data_path, params_set["vocab_size"],params_set["max_len"])

    train_text,train_label,test_text,test_label = data_prep_ELMo(x_train,y_train,x_test,y_test,params_set["max_len"])

    model = Classification_model_with_ELMo(train_text,train_label,test_text,test_label)
    print(model.summary())
