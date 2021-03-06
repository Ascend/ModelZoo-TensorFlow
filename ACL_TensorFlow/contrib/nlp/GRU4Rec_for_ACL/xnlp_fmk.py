#!/usr/bin/python3
# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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


import datetime
import os

import tensorflow.compat.v1 as tf
from absl import flags

from libs.convert_om import convert_om
from libs.convert_pb import convert_pb
from libs.convert_pbtxt import convert_pbtxt
from libs.postprocess import postprocess
from libs.predict_om import npu_predict
from libs.predict_pb import pb_predict
from libs.preprocess import preprocess
from network.run_albert_afqmc import AfqmcProcessor as AlbertAfqmc
from network.run_albert_cmnli import CmnliProcessor as AlbertCmnli
from network.run_albert_csl import CslProcessor as AlbertCsl
from network.run_albert_iflytek import IflytekProcessor as AlbertIflytek
from network.run_albert_race import RaceProcessor as AlbertRace
from network.run_albert_squad import SquadProcessor as AlbertSquad
from network.run_albert_wsc import WscProcessor as AlbertWsc
from network.run_bert_cola import ColaProcessor as BertCola
from network.run_bert_mnli import MnliProcessor as BertMnli
from network.run_bert_mrpc import MrpcProcessor as BertMrpc
from network.run_bert_ner import NerProcessor as BertNer
from network.run_bert_squad import SquadProcessor as BertSquad
from network.run_bert_tnews import TnewsProcessor as BertTnews
from network.run_biobert_ner import NerProcessor as BioNer
from network.run_biobert_re import ReProcessor as BioRe
from network.run_lstm_imdb import ImdbProcessor as LstmImdb
from network.run_roberta_lcqmc import LcqmcProcessor as RoLcqmc
from network.run_transformer_translation import Wmt32kProcessor as T2TWmt
from network.run_gru_bp import BpProcessor as Gru4Rec

FLAGS = flags.FLAGS

MODEL = {
    "albert_en": {
        # ALBERT by google
        "cola": BertCola,
        "mnli": BertMnli,
        "mrpc": BertMrpc,
        "race": AlbertRace,
        "squad": AlbertSquad,
    },
    "albert_zh": {
        # ALBERT ZH by brightmart
        "afqmc": AlbertAfqmc,
        "cmnli": AlbertCmnli,
        "csl": AlbertCsl,
        "iflytek": AlbertIflytek,
        "tnews": BertTnews,
        "wsc": AlbertWsc,
    },
    "bert": {
        # BERT by google
        "cola": BertCola,
        "mnli": BertMnli,
        "mrpc": BertMrpc,
        "ner": BertNer,
        "squad": BertSquad,
        "tnews": BertTnews,
    },
    "lstm": {
        # LSTM by HqWei
        "imdb": LstmImdb,
    },
    "transformer": {
        # transformer by tensor2tensor
        "translation": T2TWmt,
    },
    "biobert": {
        # BioBert by dmis-lab
        "ner": BioNer,
        "re": BioRe,
    },
    "roberta": {
        # RoBERTa by brightmart
        "lcqmc": RoLcqmc,
    },
    "gru": {
        # Gru4Rec by Weiping Sone
        "rec": Gru4Rec,
    },
}

# ????????????
flags.DEFINE_string(
    "action_type", None,
    "action type preprocess:   Convert dataset file(TXT or CSV or TSV or JSON) to bin file."
    "action type freeze:       Convert checkpoint files to pb model."
    "action type pbtxt:        Convert pb model to pbtxt model."
    "action type atc:          Convert pb model to om model."
    "action type npu:          Run predict using om model on npu."
    "action type cpu:          Run predict using pb model on cpu."
    "action type postprocess:  Run calculate accuracy script."
)

# ????????????
flags.DEFINE_string(
    "model_name", None,
    "The name of the model to run. Support BERT, ALBERT_EN, ..."
)

# ??????????????????
flags.DEFINE_string(
    "task_name", None,
    "The name of the task to run. Support CoLA, MNLI, MRPC, ..."
)

# checkpoints????????????
flags.DEFINE_string(
    "checkpoint_dir", None,
    "Checkpoint of fine-tuned model."
)

# ???????????????
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. "
    "In preprocess task, should contain the .tsv files (or other data files)."
    "In predict task, should contain the .bin files. "
)

# ????????????
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the output files will be written."
)

# pb????????????
flags.DEFINE_string(
    "pb_model_file", None,
    "Pb model file used to do predict."
)

# om????????????
flags.DEFINE_string(
    "om_model_file", None,
    "Om model file used to do predict."
)

# BERT ??????????????????
# ??????????????????
flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on."
)

# spm??????????????????
flags.DEFINE_string(
    "spm_model_file", None,
    "The model file for sentence piece tokenization."
)

# BERT??????????????????
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture."
)

# ?????????????????????
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased models and False for cased models."
)

# ??????Batch Size
flags.DEFINE_integer(
    "predict_batch_size", 1,
    "Total batch size for output."
)

# ??????????????????
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than this will be padded."
)

# ??????????????????
flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks."
)

# ??????????????????
flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length."
)

# ??????????????????
flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another."
)

# ??????????????????
flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file."
)

# ???????????????????????????
flags.DEFINE_string(
    "classifier_type", "softmax",
    "Type of classifier in ner task, support softmax and crf"
)

# RNN Layer??????
flags.DEFINE_integer(
    "rnn_layers", 1,
    "RNN layers numbers."
)

# RNN Size
flags.DEFINE_integer(
    "rnn_size", 100,
    "RNN size numbers."
)

# hidden_act
flags.DEFINE_string(
    "hidden_act", "tanh",
    "hidden_act type"
)

# RNN Size
flags.DEFINE_integer(
    "n_items", 37483,
    "RNN size numbers."
)

# ATC????????????????????????
# ????????????
flags.DEFINE_integer(
    "framework", 3,
    "Framework type. 0:Caffe; 1:MindSpore; 3:Tensorflow; 5:Onnx."
)

# ????????????
flags.DEFINE_string(
    "soc_version", "Ascend310",
    "The soc version."
)

# ??????????????????
flags.DEFINE_string(
    "in_nodes", None,
    "Shape of input data."
    "Separate multiple nodes with semicolons. Use double quotation marks to enclose each argument."
)

# ??????????????????
flags.DEFINE_string(
    "input_format", None,
    "Format of input data."
)

# ??????????????????
flags.DEFINE_string(
    "out_nodes", None,
    "Output nodes designated by users. "
    "Separate multiple nodes with semicolons.Use double quotation marks to enclose each argument."
)

# ???????????????????????????AUTO TUNE
flags.DEFINE_string(
    "auto_tune_mode", None,
    "Set tune mode. E.g.: GA,RL, support configure multiple, spit by ,"
)

# ????????????/????????????
flags.DEFINE_string(
    "op_select_implmode", None,
    "Set op select implmode. Support high_precision, high_performance. default: high_performance."
)

# ??????????????????
flags.DEFINE_string(
    "precision_mode", None,
    "precision mode, support force_fp16(default), allow_mix_precision, allow_fp32_to_fp16, must_keep_origin_dtype."
)

# ??????????????????
flags.DEFINE_string(
    "output_type", "FP32",
    "Set net output type. Support FP32, FP16, UINT8. E.g.: FP16, indicates that all out nodes are set to FP16."
)

# ??????keep dtype
flags.DEFINE_string(
    "keep_dtype", None,
    "Set net output type. Support FP32, FP16, UINT8. E.g.: FP16, indicates that all out nodes are set to FP16."
)

# ????????????????????????
flags.DEFINE_bool(
    "merge_input", True,
    "Merge input by batch size in npu predict."
)


def main(_):
    model_name = FLAGS.model_name.lower()
    if model_name not in MODEL.keys():
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "E", "Model: %s is not supported" % model_name))
        exit(1)

    task_name = FLAGS.task_name.lower()
    if task_name not in MODEL[model_name].keys():
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "E", "Task: %s is not supported" % task_name))
        exit(1)

    processor = MODEL[model_name][task_name]()

    action_type = FLAGS.action_type.lower()

    # ?????????????????????
    if action_type == "preprocess":
        # ?????????????????????????????????BIN??????
        preprocess(processor)
    elif action_type == "freeze":
        # checkpoints????????????????????????PB??????
        convert_pb(processor)
    elif action_type == "pbtxt":
        convert_pbtxt()
    elif action_type == "atc":
        # ATC????????????????????????OM??????
        infer_param = {
            "framework": FLAGS.framework,
            "model": FLAGS.pb_model_file,
            "output": ".".join(FLAGS.om_model_file.split('.')[:-1]),
            "out_nodes": FLAGS.out_nodes,
            "soc_version": FLAGS.soc_version,
            "input_shape": FLAGS.in_nodes,
            "auto_tune_mode": FLAGS.auto_tune_mode,
            "op_select_implmode": FLAGS.op_select_implmode,
            "precision_mode": FLAGS.precision_mode,
            "input_format": FLAGS.input_format,
            "keep_dtype": FLAGS.keep_dtype,
        }
        convert_om(infer_param)
    elif action_type == "npu":
        # ACL???????????????????????????XACL_FMK??????xacl_fmk??????????????????????????????????????????????????????
        if FLAGS.model_name.lower() in ("lstm", "transformer"):
            ids_path = os.path.join(FLAGS.data_dir, 'input_ids')
            inputs = "%s" % ids_path
        elif FLAGS.model_name.lower() in ("gru"):
            ids_path = os.path.join(FLAGS.data_dir, 'input')
            state_path = os.path.join(FLAGS.data_dir, 'rnn_state')
            inputs = "%s,%s" % (ids_path, state_path)
        else:
            # bert or albert
            ids_path = os.path.join(FLAGS.data_dir, 'input_ids')
            mask_path = os.path.join(FLAGS.data_dir, 'input_mask')
            segment_path = os.path.join(FLAGS.data_dir, 'segment_ids')
            inputs = "%s,%s,%s" % (ids_path, mask_path, segment_path)
        npu_predict(inputs)
    elif action_type == "cpu":
        pb_predict()
    elif action_type == "postprocess":
        # ??????????????????????????????
        postprocess(processor)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.FATAL)
    tf.app.run()
