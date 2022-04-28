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
import tensorflow as tf
from tensorflow.python.framework import graph_util
from utils import dataset, utils
import argparse
import os, tqdm
import numpy as np
import torch
from utils.evaluator import CZSL_Evaluator
from collections import defaultdict
from npu_bridge.npu_init import *


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output", required=True,
                        help="output path")
    parser.add_argument("--data_url", type=str, default="./data", required=True,
                        help="input path")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, default='UT', choices=['MIT', 'UT', 'MITg', 'UTg'], help="Dataset name")
    parser.add_argument("--test_bz", type=int, default=1024, help="Test batch size")
    parser.add_argument("--obj_pred", type=str, default=None, help="Object prediction from pretrained model")
    return parser


def freeze_graph(input_checkpoint, output_graph):
    """
    ckpt转pb
    """
    # 输出节点
    output_node_names = "Mul_18,Softmax_3,Placeholder_6"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with utils.create_session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))  # 多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))


def test_pb(args, pb_path):
    """
    在线推理，测试pb模型
    """
    print("Loading test dataset")
    test_dataloader = dataset.get_dataloader(args.data_url, args.data, 'test', batchsize=args.test_bz,
                                             obj_pred=args.obj_pred)

    network = 1
    evaluator = CZSL_Evaluator(test_dataloader.dataset, network)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with utils.create_session() as sess:
            sess.run(tf.global_variables_initializer())

            print("get input tensor")
            # 输入张量
            pos_image_feat = sess.graph.get_tensor_by_name("Placeholder_2:0")
            test_attr_id = sess.graph.get_tensor_by_name("test_attr_id:0")
            test_obj_id = sess.graph.get_tensor_by_name("test_obj_id:0")
            pos_obj_prediction = sess.graph.get_tensor_by_name("Placeholder_6:0")

            print("get output tensor")
            # 输出张量
            prob_P_rmd = sess.graph.get_tensor_by_name("Mul_18:0")
            prob_A_attr = sess.graph.get_tensor_by_name("Softmax_3:0")
            prob_O = sess.graph.get_tensor_by_name("Placeholder_6:0")
            score_op = dict([
                ("score_rmd", [prob_P_rmd, prob_A_attr, prob_O]),  # Mul_18, Softmax_3, Placeholder_6
            ])

            accuracies_pair = defaultdict(list)
            accuracies_attr = defaultdict(list)
            accuracies_obj = defaultdict(list)

            for image_ind, batch in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), postfix='test'):
                dset = test_dataloader.dataset
                test_att = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
                test_obj = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])

                feed_dict = {
                    pos_image_feat: batch[4],  # Placeholder_2
                    test_attr_id: test_att,  # test_attr_id
                    test_obj_id: test_obj,  # test_obj_id
                    pos_obj_prediction: batch[-1],  # Placeholder_6
                }
                score = sess.run(score_op, feed_dict=feed_dict)
                for key in score_op.keys():
                    score[key][0] = {
                        (a, o): torch.from_numpy(score[key][0][:, i])
                        for i, (a, o) in enumerate(zip(test_att, test_obj))
                    }

                attr_truth, obj_truth = batch[1], batch[2]
                attr_truth, obj_truth = torch.from_numpy(attr_truth), torch.from_numpy(obj_truth)

                match_stats = []
                for key in score_op.keys():
                    p_pair, p_a, p_o = score[key]
                    pair_results = evaluator.score_model(p_pair, obj_truth)
                    match_stats = evaluator.evaluate_predictions(pair_results, attr_truth, obj_truth)
                    accuracies_pair[key].append(match_stats)  # 0/1 sequence of t/f

                    a_match, o_match = evaluator.evaluate_only_attr_obj(p_a, attr_truth, p_o, obj_truth)

                    accuracies_attr[key].append(a_match)
                    accuracies_obj[key].append(o_match)

            for name in accuracies_pair.keys():
                accuracies = accuracies_pair[name]
                accuracies = zip(*accuracies)
                accuracies = map(torch.mean, map(torch.cat, accuracies))
                attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, objoracle_acc = map(lambda x: x.item(),
                                                                                                    accuracies)

                real_attr_acc = torch.mean(torch.cat(accuracies_attr[name])).item()
                real_obj_acc = torch.mean(torch.cat(accuracies_obj[name])).item()

                report_dict = {
                    'real_attr_acc': real_attr_acc,
                    'real_obj_acc': real_obj_acc,
                    'top1_acc': closed_1_acc,
                    'top2_acc': closed_2_acc,
                    'top3_acc': closed_3_acc,
                    'name': "symnet",
                    'epoch': 636,
                }

                print(name + ": " + utils.formated_czsl_result(report_dict))

    pass


def main():
    parser = make_parser()
    args = parser.parse_args()

    weight_dir = os.path.join(args.data_url, './weights')
    ckpt_path = os.path.join(weight_dir, args.ckpt)
    print("ckpt path    => ", ckpt_path)

    pb_path = os.path.join(args.train_url, './pb/')
    print("pb path      => ", pb_path)
    if not os.path.exists(pb_path):
        os.mkdir(pb_path)

    saved_pb_path = os.path.join(args.data_url, './pb/symnet.pb')
    print("saved pb path    => ", saved_pb_path)

    freeze_graph(ckpt_path, pb_path + "symnet.pb")
    # test_pb(args, pb_path=pb_path+"symnet.pb")


if __name__ == '__main__':
    main()
