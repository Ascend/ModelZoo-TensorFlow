import argparse
import os
import sys
import time

import tensorflow as tf
from absl import logging
slim = tf.contrib.slim


def main(config):
    mask_neg = tf.constant([[True],[False], [True], [False]])
    b = tf.constant([3.,4.,5.,6.])
    # c = tf.multiply(a,b)
    loss_class = tf.where_v2(mask_neg, 1 - tf.expand_dims(b, -1), 0)
    with tf.Session(config=config) as sess:
        c_ = sess.run(loss_class)
        print(c_)

if __name__ == '__main__':
    # npu setting
    import npu_bridge
    import moxing as mox
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    os.environ['PRINT_MODEL'] = '1'
    os.environ['SLOG_PRINT_TO_STDOUT'] = "1"
    os.environ['DUMP_GE_GRAPH'] = "2"
    os.environ['DUMP_GRAPH_LEVEL'] = "3"
    os.environ['ENABLE_NETWORK_ANALYSIS_DEBUG'] = "1"
    main(config)