#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File       : main.py
@Time       : 2022/12/12/14:45
@Author     : 曹禹臣
@Software   : Pycharm
@Description:
"""
import argparse
import os
from collections import OrderedDict

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_text

# 在使用t5时需要使用 tensorflow_text 注册一些算子，不导入该萨嘎无法有问题
# 在推理代码中，虽并未明文使用，但是不加这一句会因为代码格式化因模块未使用成果被优化掉，故加上这一句更好
_ = tensorflow_text


def get_config():
    import npu_device
    from npu_device.compat.v1.npu_init import RewriterConfig
    npu_device.compat.enable_v1()
    config = tf1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    return config


class T5:
    def __init__(self, pb_file, config):
        graph = tf.Graph()
        with graph.as_default():
            with tf.io.gfile.GFile(pb_file, 'rb') as f:
                graph_def = tf1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        self.graph = graph
        self.config = config

    def predict(self, questions):
        results = OrderedDict()
        with tf1.Session(graph=self.graph, config=self.config) as sess:
            out_nodes = ['strided_slice_1:0', 'strided_slice_2:0']
            input_node = self.graph.get_tensor_by_name('inputs:0')
            for question in questions:
                feed_dict = {input_node: [question]}
                results[question] = sess.run(out_nodes, feed_dict=feed_dict)
        return results

    def answer(self, questions):
        return OrderedDict([(k, v[1][0].decode('utf8')) for k, v in self.predict(questions).items()])


def parse_args():
    parser = argparse.ArgumentParser(description='t5 runner')
    parser.add_argument('-m', '--model', '--model-path', dest='model_path', type=str, default='model/t5.pb',
                        help='model路径')
    parser.add_argument('-f', '--file', dest='question_file', type=str, default='',
                        help='问题所在文本文件，以\\n分隔，如果此参数有值，则优先使用此参数')
    parser.add_argument(nargs='*', dest='questions', default=list(), help='问题列表')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.exists(args.question_file):
        with open(args.question_file, encoding='utf8') as f:
            questions = [question for question in f.read().split('\n') if question]
    else:
        questions = args.questions

    assert questions, '请提供问题'

    t5 = T5(args.model_path, config=get_config())
    result_dict = t5.answer(questions)
    print('-' * 50)
    for question, answer in result_dict.items():
        print(f"{question}:\n    {answer}")
    print('=' * 50)


if __name__ == '__main__':
    main()
