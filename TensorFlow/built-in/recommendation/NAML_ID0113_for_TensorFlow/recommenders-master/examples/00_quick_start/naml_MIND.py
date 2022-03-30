#! -*- coding:utf-8 -*-

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. 
# =======================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import tensorflow as tf
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops
from hccl.manage.api import get_rank_size, get_rank_id
# from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator
# from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
from npu_bridge.estimator.npu import npu_convert_dropout


def main():

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    seed = 42

    args = parse_args()

    train_news_file = os.path.join(args.data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(args.data_path, 'train', r'behaviors.tsv')
    valid_news_file = os.path.join(args.data_path, 'valid', r'news.tsv')
    valid_behaviors_file = os.path.join(args.data_path, 'valid', r'behaviors.tsv')
    wordEmb_file = os.path.join(args.data_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(args.data_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(args.data_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(args.data_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(args.data_path, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(args.data_path, "utils", r'naml.yaml')

    # mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(args.MIND_type)
    #
    # if not os.path.exists(train_news_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'train'), mind_train_dataset)
    #
    # if not os.path.exists(valid_news_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'valid'), mind_dev_dataset)
    # if not os.path.exists(yaml_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'utils'), mind_utils)

    # ++++++++++++++++++++++++++++++++ modify for 8p +++++++++++++++++++++++++++++++++++++++++++++++++
    npu_init = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = False
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    json_path = os.path.dirname(os.path.realpath(__file__))
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(os.path.join(json_path, "ops_info.json"))
    rank_size = os.environ.get('RANK_SIZE', '').strip()
    if int(rank_size) > 1:
        custom_op.parameter_map["hcom_parallel"].b = True
    
    custom_op.parameter_map["dynamic_input"].b = True
    if args.dynamic_input == "lazy_recompile":
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    elif args.dynamic_input == "1":
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
    else:
        print("Enter correct compilation parameters.")

    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.compat.v1.Session(config=sess_config)

    # set this TensorFlow session as the default session for Keras
    tf.compat.v1.keras.backend.set_session(sess)
    sess.run(npu_init)
    #rank_size = get_rank_size()
    #rank_id = get_rank_id()
    rank_size = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv("RANK_ID"))
    
    # ++++++++++++++++++++++++++++++++ end for 8P ++++++++++++++++++++++++++++++++++++++++++++++++++++

    hparams = prepare_hparams(yaml_file,
                              wordEmb_file=wordEmb_file,
                              wordDict_file=wordDict_file,
                              userDict_file=userDict_file,
                              vertDict_file=vertDict_file,
                              subvertDict_file=subvertDict_file,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              max_steps=args.max_steps,
                              model_path=args.model_path,
                              rank_size=rank_size,
                              rank_id=rank_id)
    print(hparams)

    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator, seed=seed)

    os.makedirs(args.model_path, exist_ok=True)

    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

    #res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
    #print(res_syn)
    #sb.glue("res_syn", res_syn)

    sess.run(npu_shutdown)
    sess.close()


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="""batchsize""")
    parser.add_argument('--epochs', type=int, default=1,
                        help="""epoch""")
    parser.add_argument('--model_path', type=str, default='./',
                        help="""pb path""")
    parser.add_argument('--data_path', type=str, default='./',
                        help = """the preprocess path of output data""")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="""the max train steps""")
    parser.add_argument('--MIND_type', default='small', choices=["demo", "small", "large"],
                        help = """the type of MIND data""")
    parser.add_argument('--dynamic_input', type=str, default='1',
                        help="--dynamic_input=1 Use fuzzy compilation. --dynamic_input=lazy_recompile Compile using lazy static graph")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

if __name__ == '__main__':
    main()
