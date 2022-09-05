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
#
import npu_device
# from resnet_groupNorm import train_model
from resnet_batchRenorm import train_model
# from resnet import train_model
import tensorflow as tf
import os, time, argparse, ast  

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="ckpt", type=str)
parser.add_argument("--log_dir", default="log", type=str)
parser.add_argument("--data_path", help="data_path", type=str)
parser.add_argument("--steps", type=int, help="steps")
parser.add_argument("--epochs", type=int, help="epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
#===============================NPU Migration=========================================
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')
############多p参数##############
parser.add_argument("--rank_size", default=1, type=int, help="rank size")
parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")
args = parser.parse_args()

batch_size = args.batch_size
batch_multiplier = 6
reg_coef = 1.0
def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()

npu_config(args)

def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int32)
    return img, label


dataset = tf.data.TFRecordDataset(os.path.join(args.data_path, 
    'dataset/converted_dataset/ms1m_train.tfrecord'))
if args.rank_size !=1:
    dataset, batch_size = npu_device.distribute.shard_and_rebatch_dataset(dataset, args.batch_size)
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(buffer_size=20000)
dataset = dataset.batch(batch_size * batch_multiplier)

print("Preparing model...")

model = train_model()

learning_rate = 0.0005

if args.rank_size !=1:
    optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.SGD(
        lr=learning_rate, momentum=0.9, nesterov=False))
else:
    optimizer = tf.keras.optimizers.SGD(
        lr=learning_rate, momentum=0.9, nesterov=False)

# optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
# optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=0.0)
if args.rank_size !=1:
    training_vars = model.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank=0)

@tf.function
def train_step(images, labels, regCoef):
    # print(images, labels)
    with tf.GradientTape() as tape:
        logits = model(tf.slice(images, [0, 0, 0, 0], [
                       batch_size, 112, 112, 3]), tf.slice(labels, [0], [batch_size]))
        for i in range(batch_multiplier - 1):
            logits = tf.concat([logits, model(tf.slice(images, [batch_size * (i + 1), 0, 0, 0], [
                               batch_size, 112, 112, 3]), tf.slice(labels, [batch_size * (i + 1)], [batch_size]))], 0)
        pred = tf.nn.softmax(logits)
        inf_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        reg_loss = tf.add_n(model.losses)
        loss = inf_loss + reg_loss * regCoef
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss = tf.reduce_mean(loss)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.dtypes.int32), labels), dtype=tf.float32))
    inference_loss = tf.reduce_mean(inf_loss)
    regularization_loss = tf.reduce_mean(reg_loss)
    return accuracy, train_loss, inference_loss, regularization_loss


EPOCHS = args.epochs

# create log
summary_writer = tf.summary.create_file_writer(args.log_dir)

lr_steps = [int(40000 * 512 / (batch_size * batch_multiplier)),
            int(60000 * 512 / (batch_size * batch_multiplier)),
            int(80000 * 512 / (batch_size * batch_multiplier)),
            int(120000 * 512 / (batch_size * batch_multiplier))]
print(lr_steps)
step = 0
startTime = None
for epoch in range(EPOCHS):
    iterator = iter(dataset)
    while True:
        img, label = next(iterator)
        if (img.shape[0] != batch_size * batch_multiplier or img.shape[0] != label.shape[0] or step >= args.steps):
            print("End of epoch {}".format(epoch + 1))
            break
        step += 1
        accuracy, train_loss, inference_loss, regularization_loss = train_step(
            img, label, reg_coef)
        if step == 1:
           startTime = time.time()
        if step % 200 == 0:
            endTime = time.time()
            template = 'Epoch {}, Step {}, Loss: {}, Reg loss: {}, Accuracy: {}, Reg coef: {}, step/sec: {}, FPS: {}'
            print(template.format(epoch + 1, step,
                                  '%.5f' % (inference_loss),
                                  '%.5f' % (regularization_loss),
                                  '%.5f' % (accuracy),
                                  '%.5f' % (reg_coef),
                                  '%.5f' % ((endTime - startTime) / (step - 1)),
                                  '%.5f' % (batch_size / ((endTime - startTime) / (step - 1)))
                                  ))
            # with summary_writer.as_default():
            #     tf.summary.scalar(
            #         'train loss', train_loss, step=step)
            #     tf.summary.scalar(
            #         'inference loss', inference_loss, step=step)
            #     tf.summary.scalar(
            #         'regularization loss', regularization_loss, step=step)
            #     tf.summary.scalar(
            #         'train accuracy', accuracy, step=step)
            #     tf.summary.scalar(
            #         'learning rate', optimizer.lr, step=step)
                # for i in range(len(gradients)):
                #     gradient_name = model.trainable_variables[i].name
                #     tf.summary.histogram(
                #         gradient_name + '/gradient', gradients[i], step=step)
                # for weight in model.trainable_variables:
                #     tf.summary.histogram(
                #         weight.name, weight, step=step)
                # layer_output = model.get_layer('').output
                # tf.summary.histogram('name', layer_output)
        if (step % 4000 == 0 and step > 0) or step == args.steps:
            model.save_weights(os.path.join(args.ckpt_dir,
                'weights_epoch-{}_step-{}'.format(EPOCHS, step)))
        for lr_step in lr_steps:
            if lr_step == step:
                optimizer.lr = optimizer.lr * 0.5
        if inference_loss * 1.0 < regularization_loss * reg_coef:
            reg_coef = reg_coef * 0.8
