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
import numpy as np

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def logits_2_pixel_value(logits, mu=1.1):
  """
  Args:
    logits: [n, 256] float32
    mu    : float32
  Returns:
    pixels: [n] float32
  """
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, 256, dtype=np.float32)
  pixels = np.sum(probs * pixel_dict, axis=1)
  return np.floor(pixels)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def save_samples(np_imgs, img_path):
  """
  Args:
    np_imgs: [N, H, W, 3] float32
    img_path: str
  """
  np_imgs = np_imgs.astype(np.uint8)
  N, H, W, _ = np_imgs.shape
  num = int(N ** (0.5))
  merge_img = np.zeros((num * H, num * W, 3), dtype=np.uint8)
  for i in range(num):
    for j in range(num):
      merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]

  # imsave(img_path, merge_img)
  # misc.imsave(img_path, merge_img)
  #------------------NPU 2021.10.17-------------------------
  img = Image.fromarray(np.uint8(merge_img))
  img.save(img_path)
  #------------------NPU 2021.10.17-----------------

pb_path = "./pb_model/prsr_conditioning.pb"

dataset = DataSet("./train.txt", 30)
lr_imgs = dataset.lr_images
hr_imgs = dataset.hr_images
gen_hr_imgs = np.zeros((1, 32, 32, 3), dtype = np.float32)

with tf.gfile.FastGFile(pb_path, 'rb') as model_file:
    with tf.Session() as sess:
        np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
        graph = tf.Graph().as_default()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        tf.import_graph_def(graph_def,name="")
        input1 = sess.graph.get_tensor_by_name("input1:0")
        input2 = sess.graph.get_tensor_by_name("input2:0")
        c_logits = sess.graph.get_tensor_by_name("c_logits:0")#自己定义的输出tensor名称
        p_logits = sess.graph.get_tensor_by_name("p_logits:0")#自己定义的输出tensor名称

        ret = sess.run([c_logits, p_logits], feed_dict={input1:hr_imgs.eval(), input2:lr_imgs.eval()})
        # print(ret[0])
        c_logic = ret[0]
        p_logic = ret[1]

        mu = 1.0
        for i in range(32):
            for j in range(32):
                for c in range(3):
                    new_pixel = logits_2_pixel_value(
                            c_logic[:, i, j, c * 256:(c + 1) * 256] + p_logic[:, i, j, c * 256:(c + 1) * 256],
                            mu = mu)
                    gen_hr_imgs[:, i, j, c] = new_pixel

save_samples(gen_hr_imgs, './outimg/pb_generate_imgs' + '.jpg')
save_samples(np_hr_imgs, './outimg/pb_hr_imgs' + '.jpg')