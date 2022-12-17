import tensorflow as tf
from tensorflow.python.framework import graph_util
import model as model_utils

inputs = tf.placeholder(tf.float32, shape=[64, 128, 128, 3],name='input')  # input shape
    # create inference graph
resolution = (128, 128)
batch_size=64
num_slots=7
num_iterations=3

model = model_utils.build_model(resolution, batch_size, num_slots,
                                num_iterations, model_type="object_discovery")
logit1,logit2,logit3,logit4 = model(inputs, training=False)
print("-----------------------------------------测试完成-------------------------------------------")
saver = tf.train.Saver(max_to_keep=5)
# graph_def = tf.get_default_graph().as_graph_def()

with tf.Session() as sess:
    saver.restore(sess, '/home/disk/checkp/checkpoint.ckpt-499000')
    print("---------------------开始转换-------------------------------")
    output_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph_def,['model/slot_attention_auto_encoder/Sum'])
    print("---------------------转换完成--------------------------")
    # print(sess.run(tf.get_default_graph().get_tensor_by_name('model/slot_attention_auto_encoder/Sum:0')))  # 3.0
    with tf.gfile.GFile('./newslotmodel.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())  # 得到文件：model.pb
