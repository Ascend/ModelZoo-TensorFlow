import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy
# from keras.applications.xception import preprocess_input



config = tf.ConfigProto()
sess = tf.Session(config=config)
with gfile.FastGFile('./placeholder_protobuf.pb', 'rb') as f:
     graph_def = tf.GraphDef()
     graph_def.ParseFromString(f.read())
     sess.graph.as_default()
     tf.import_graph_def(graph_def, name='')

print('----------')
 # 获取输入tensor
taskdata = tf.get_default_graph().get_tensor_by_name("taskdata:0")
taskdata_1 = tf.get_default_graph().get_tensor_by_name("taskdata_1:0")
taskdata_2 = tf.get_default_graph().get_tensor_by_name("taskdata_2:0")
print("input:", taskdata)
# 获取预测tensor
pred = tf.get_default_graph().get_tensor_by_name("policy/Reshape_3:0")  # mobilenet_v2
print(pred)

x1 = np.fromfile('./taskdata_0.bin', dtype=np.float32)
x2 = np.fromfile('./taskdata_1.bin', dtype=np.float32)
x3 = np.fromfile('./taskdata_2.bin', dtype=np.float32)

x1=x1.reshape([8, 20, 64, 64, 90])
x2=x2.reshape([8, 20, 5])
x3=x3.reshape([8, 20, 8])

res = sess.run(pred, feed_dict={taskdata: x1, taskdata_1: x2, taskdata_2: x3})

print(res.shape)
print(res)
