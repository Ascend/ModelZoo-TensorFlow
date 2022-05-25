import tensorflow as tf
from vit_keras import vit, utils
import os
import time
import tensorflow.keras.backend as K
from PIL import Image
import pickle
import numpy as np
import argparse

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from npu_bridge.npu_init import *
######################
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import ProfilingConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets', help='Datasets location')
    parser.add_argument('--output_path', type=str, default='./output', help='Output location,saving trained models')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='learning rate')
    return parser.parse_args()

args = parse_args()



MODEL_CACHE_PATH=args.output_path
PRETRAINED_MODEL_PATH= os.path.join(args.data_path,"model")
HPARAMS = {
    "batch_size": 4,
    "image_size": 384,
    'learning_rate': 0.001,
}
DATA_CACHE_PATH= args.data_path
image_size = HPARAMS['image_size']
batch_size = HPARAMS['batch_size']

def read_data(filename, training):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if training:
        images = dict[b'data'].reshape([50000, 3, 32, 32])
    else:
        images = dict[b'data'].reshape([10000, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(dict[b'fine_labels'])
    def _augment(image, label):
        if np.random.rand() < 0.3:
            image = tf.image.flip_left_right(image)
        if np.random.rand() < 0.3:
            image = tf.image.flip_up_down(image)
        if np.random.rand() < 0.3:
            image = tf.image.random_contrast(image, lower=0.5, upper=2)
        return image, label

    def _preprocess(image, label):
        image = tf.image.resize(image, (image_size, image_size))
        image = (image - 127.5) / 127.5
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        # ds = ds.map(_augment)
        ds = ds.map(_preprocess)
        ds = ds.shuffle(HPARAMS['batch_size'] * 10)
        ds = ds.repeat()
    else:
        ds = ds.map(_preprocess)
        ds = ds.repeat()

    ds = ds.batch(batch_size=HPARAMS['batch_size'], drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    image_batch, label_batch = iterator.get_next()
    print("load dataset =================================")
    return image_batch, label_batch


model = vit.vit_b16_load_pretrain(
    image_size=HPARAMS['image_size'],
    activation='linear',
    pretrained=True,
    classes=100,
    include_top=True,
    pretrained_top=True,
    pretrained_path="./"
)

images_batch, labels_batch = read_data(filename=os.path.join(DATA_CACHE_PATH,"train"), training=True)
val_image_batch, val_labels_batch = read_data(filename=os.path.join(DATA_CACHE_PATH,"test"), training=False)

loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

inputx = tf.compat.v1.placeholder(
    tf.float32, shape=[HPARAMS['batch_size'], HPARAMS['image_size'], HPARAMS['image_size'], 3], name="inputx")

inputy = tf.compat.v1.placeholder(
    tf.int64, shape=[HPARAMS['batch_size'], ], name="inputy")

inputTrain = tf.compat.v1.placeholder(
    tf.bool, name='training')


def eval(pred, label):
    prediction = np.argmax(pred, 1).tolist()
    return calc(prediction, label)


def calc(prediction, label):
    a = [prediction[i] == label[i] for i in range(len(prediction))]
    return sum(a) / len(a)


out = model(inputx, training=inputTrain)
loss = loss_fun(inputy, out)

optimizer = tf.train.MomentumOptimizer(
    learning_rate=HPARAMS['learning_rate'],
    momentum=0.9,
    use_locking=False,
    use_nesterov=False,
    name='Momentum'
)



loss_scale_manager = FixedLossScaleManager(loss_scale=2**32)

opt = NPULossScaleOptimizer(optimizer, loss_scale_manager)


train_op = opt.minimize(loss)


config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s=tf.compat.as_bytes("allow_mix_precision")
custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./fusion_switch.cfg")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
config.gpu_options.allow_growth = True

session_config = config
sess = tf.Session(config=session_config)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver(max_to_keep=10)
s = time.time()
print("============ start load pretrained model =======================================")
saver.restore(sess, "{}/vit-base-5".format(PRETRAINED_MODEL_PATH))  # 0.8709  for 5   # 2 0.8719  # 3 0.8767
print("============ load success {:.4f} =====================".format(time.time() - s))

for epoch in range(1, 2):
    # train
    label_col = []
    pred_col = []
    for step in range(HPARAMS['batch_size']*1000 // HPARAMS['batch_size']):
        s = time.time()
        x_in, y_in = sess.run([images_batch, labels_batch],
                              feed_dict={inputTrain: True})
        out_, loss_, _ = sess.run([out, loss, train_op], feed_dict={
            inputx: x_in, inputy: y_in, inputTrain: True})
        label_col += y_in.tolist()
        pred_col += np.argmax(out_, 1).tolist()
        if step!=0:
            print("epoch:{}  step: {} , loss: {:.4f} ,  sec/step : {:.4f}  acc: {:.4f}".format(
                    epoch, step, loss_.item(), time.time() - s, eval(out_, y_in))) 

