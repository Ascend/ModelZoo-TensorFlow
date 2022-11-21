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
from npu_bridge.npu_init import *
import json
import fasttext
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model
from keras.datasets import imdb
from gensim.models import Word2Vec
from gensim.models import FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Embedding,Dense,Flatten
from sklearn.metrics import accuracy_score,classification_report

#import tensorflow as tf
import tensorflow.python.keras as keras
#from tensorflow.python.keras import backend as K
from keras import backend as K
from npu_bridge.npu_init import *

def npu_keras_optimizer(opt):
    npu_opt = KerasDistributeOptimizer(opt)
    return npu_opt

def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set


with open("model_params.json", "r") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)


def load_data(datapath,vocab_size,max_len):
    """
        Loads the keras imdb dataset

        Args:
            vocab_size = {int} the size of the vocabulary
            max_len = {int} the maximum length of input considered for padding

        Returns:
            X_train = tokenized train data
            X_test = tokenized test data

    """
    INDEX_FROM = 3
    data_path = ('%s/keras/datasets/imdb.npz' % (datapath))
    (X_train,y_train),(X_test,y_test) = imdb.load_data(path = data_path, num_words = vocab_size,index_from = INDEX_FROM)

    return X_train,X_test,y_train,y_test


def prepare_data_for_word_vectors_imdb(data_path, X_train):
    """
        Prepares the input

        Args:
            X_train = tokenized train data

        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus

    """
    INDEX_FROM = 3
    wordpath = ('%s/keras/datasets/imdb_word_index.json' % (data_path))
    word_to_index = imdb.get_word_index(path = wordpath)
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences = []
    for i in range(len(X_train)):
        temp = [index_to_word[ids] for ids in X_train[i]]
        sentences.append(temp)
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_indexes = tokenizer.word_index
    """
    return sentences,word_to_index


def prepare_data_for_word_vectors(X):
    sentences_as_words=[]
    word_to_index={}
    count=1
    for sent in X:
        temp = sent.split()
        sentences_as_words.append(temp)
    for sent in sentences_as_words:
        for word in sent:
            if word_to_index.get(word,None) is None:
                word_to_index[word] = count
                count +=1
    index_to_word = {v:k for k,v in word_to_index.items()}
    sentences=[]
    for i in range(len(sentences_as_words)):
        temp = [word_to_index[w] for w in sentences_as_words[i]]
        sentences.append(temp)


    return sentences_as_words,sentences,word_to_index

def data_prep_ELMo(train_x,train_y,test_x,test_y,max_len):

    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences=[]
    for i in range(len(train_x)):
        temp = [index_to_word[ids] for ids in train_x[i]]
        sentences.append(temp)

    test_sentences=[]
    for i in range(len(test_x)):
        temp = [index_to_word[ids] for ids in test_x[i]]
        test_sentences.append(temp)

    train_text = [' '.join(sentences[i][:max_len]) for i in range(len(sentences))]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_y.tolist()

    test_text = [' '.join(test_sentences[i][:500]) for i in range(len(test_sentences))]
    test_text = np.array(test_text , dtype=object)[:, np.newaxis]
    test_label = test_y.tolist()

    return train_text,train_label,test_text,test_label


def building_word_vector_model(option,sentences,embed_dim,workers,window,y_train):
    """
        Builds the word vector

        Args:
            type = {bool} 0 for Word2vec. 1 for gensim Fastext. 2 for Fasttext 2018.
            sentences = {list} list of tokenized words
            embed_dim = {int} embedding dimension of the word vectors
            workers = {int} no. of worker threads to train the model (faster training with multicore machines)
            window = {int} max distance between current and predicted word
            y_train = y_train

        Returns:
            model = Word2vec/Gensim fastText/ Fastext_2018 model trained on the training corpus


    """
    if option == 0:
        print("Training a word2vec model")
        model = Word2Vec(sentences=sentences, size = embed_dim, workers = workers, window = window)
        print("Training complete")

    elif option == 1:
        print("Training a Gensim FastText model")
        model = FastText(sentences=sentences, size = embed_dim, workers = workers, window = window)
        print("Training complete")

    elif option == 2:
        print("Training a Fasttext model from Facebook Research")
        y_train = ["__label__positive" if i==1 else "__label__negative" for i in y_train]

        with open("imdb_train.txt","w") as text_file:
            for i in range(len(sentences)):
                print(sentences[i],y_train[i],file = text_file)

        model = fasttext.skipgram("imdb_train.txt","model_ft_2018_imdb",dim = embed_dim)
        print("Training complete")

    return model

def padding_input(X_train,X_test,maxlen):
    """
        Pads the input upto considered max length

        Args:
            X_train = tokenized train data
            X_test = tokenized test data

        Returns:
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data

    """

    X_train_pad = pad_sequences(X_train,maxlen=maxlen,padding="post")

    X_test_pad = pad_sequences(X_test,maxlen=maxlen,padding="post")

    return X_train_pad,X_test_pad


def ELMoEmbedding(x):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


def classification_model(args,embed_dim,X_train_pad,X_test_pad,y_train,y_test,vocab_size,word_index,w2vmodel,
                         trainable_param,option):
    """
        Builds the classification model for sentiment analysis

        Args:
            embded_dim = {int} dimension of the word vectors
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
            vocab_size = {int} size of the vocabulary
            word_index =  {dict} word and its indexes in whole of imdb corpus
            w2vmodel = Word2Vec model
            trainable_param = {bool} whether to train the word embeddings in the Embedding layer
            option = {int} choice of word embedding
    """
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
   # custom_op.parameter_map["dynamic_input"].b = True
   # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    if args.data_dump_flag.strip()=="True":
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if args.over_dump.strip()=="True":
        # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
        # enable_dump_debug：是否开启溢出检测功能
        custom_op.parameter_map["enable_dump_debug"].b = True
        # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if args.profiling.strip()=="True":
        custom_op.parameter_map["profiling_mode"].b = False
        profilingvalue=('{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' %(args.profiling_dump_path))
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)

    sess = tf.Session(config=sess_config)
    K.set_session(sess)

    embedding_matrix = np.zeros((vocab_size,embed_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = w2vmodel[word]
        except:
            pass
        try:
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
        except:
            pass

    embed_layer = Embedding(vocab_size,embed_dim,weights =[embedding_matrix],trainable=trainable_param)

    input_seq = Input(shape=(X_train_pad.shape[1],))
    embed_seq = embed_layer(input_seq)
    x = Dense(256,activation ="relu")(embed_seq)
    x = Flatten()(x)
    preds = Dense(1,activation="sigmoid")(x)

    model = Model(input_seq,preds)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    if args.loss_scale == 'True':
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32,
                                                               incr_every_n_steps=1000, decr_every_n_nan_or_inf=2,
                                                               decr_ratio=0.8)
    if int(os.getenv('RANK_SIZE')) == 1:
        optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
    else:
        optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)
    optim = npu_tf_optimizer(optimizer)

    model.compile(loss=model_params["loss"],optimizer=optim,metrics= model_params["metrics"])
    model.fit(X_train_pad,y_train,epochs=args.epoch,batch_size=args.batch_size,validation_data=(X_test_pad,y_test))
    
    print('====save model====')
    #model.save_weights('./ckpt_gpu/model_weigits.h5')
    #model.save('./ckpt_gpu/model.h5')
    predictions = model.predict(X_test_pad, batch_size=1)
    predictions = [0 if i<0.5 else 1 for i in predictions]
    print("Accuracy: ",accuracy_score(y_test,predictions))
    print("Classification Report: ",classification_report(y_test,predictions))
    sess.close()
    return model

def Classification_model_with_ELMo(X_train_pad,y_train,X_test_pad,y_test):
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embed_seq = layers.Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    x = Dense(256,activation ="relu")(embed_seq)
    preds = Dense(1,activation="sigmoid")(x)
    model = Model(input_text,preds)


    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

    model.fit(X_train_pad,y_train,epochs=10,batch_size=512,validation_data=(X_test_pad,y_test))

    predictions = model.predict(X_test_pad)
    predictions = [0 if i<0.5 else 1 for i in predictions]
    print("Accuracy: ",accuracy_score(y_test,predictions))
    print("Classification Report: ",classification_report(y_test,predictions))

    return model
