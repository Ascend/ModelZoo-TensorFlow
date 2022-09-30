import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
from tensorflow.python.framework import graph_util
from ops import *
from data import *
from net import *
from utils import *
import npu_bridge 
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig



config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

batch_size = 1

dataset = DataSet("./train.txt", 30)

ckpt_path = "./output/model.ckpt-280000"

def main():
    tf.reset_default_graph()
    inputs1 = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="input1")
    inputs2 = tf.placeholder(tf.float32, shape = [1, 8, 8, 3], name = "input2")
    net = Net(inputs1, inputs2, 'prsr')
    net.train = tf.constant(False)
    #c_logits = net.conditioning_logits
    #p_logits = net.prior_logits
    c_logits = tf.identity(net.conditioning_logits, name='c_logits')
    p_logits = tf.identity(net.prior_logits, name='p_logits')
    with tf.Session(config=config) as sess:
    
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='p_logits,c_logits',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/prsr_conditioning.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()