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

from segdec_model import SegDecModel
from segdec_data import InputData
from segdec_train import SegDecTrain
import tensorflow as tf
import numpy as np
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=str, default='0', help="Delimited list")

    args = parser.parse_args()
    folds = args.folds
    use_cross_entropy_seg_net = True
    pos_weights = 1

    net_model =SegDecModel(decision_net=SegDecModel.DECISION_NET_FULL,
                           use_corss_entropy_seg_net=use_cross_entropy_seg_net,
                           positive_weight=pos_weights)
    current_pretrained_model = './output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_'+folds
    train = SegDecTrain(net_model,
                        storage_dir='./output',
                        run_string='KolektorSDD-dilate=5/full-size_cross-entropy/fold_'+folds,
                        image_size=(1408, 512, 1),
                        batch_size=1,
                        learning_rate=0.1,
                        max_steps=660,
                        max_epochs=1200,
                        visible_device_list='0',
                        pretrained_model_checkpoint_path=current_pretrained_model,
                        train_decision_net=True,
                        train_segmentation_net=False,
                        use_random_rotation=False,
                        ensure_posneg_balance=True)

    dataset = InputData('test', './db/KolektorSDD-dilate=5/fold_'+folds)
    images, labels, img_names = train.input.add_inputs_nodes(dataset, False)
    with tf.name_scope('%s_%d' %('tower',0)) as scope:
        net, decision, _ = net_model.get_inference(images, 3, scope=scope)
    net_op = net
    decision_op = decision
    images_op = images
    labels_op = labels
    img_names_op = img_names
    c = tf.ConfigProto()
    with tf.Session(config=c) as sess:
        variable_averages = tf.train.ExponentialMovingAverage(net_model.MOVING_AVERAGE_DECAY)
        variables_to_restore =variable_averages.variables_to_restore()
        ckpt_file = './output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_'+folds+'/model.ckpt-659'
        net_model.restore(sess, ckpt_file, variables_to_restore)

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            samples_outcome = []
            sample_names = []
            samples_speed_eval = []
            step = 0

            while step < 135:

                if decision is None:
                    predictions, image, label, img_name = sess.run([net, images, labels, img_names])

                else:
                    predictions, decision, image, label,img_name = sess.run([net_op,decision_op, images_op, labels_op, img_names_op])
                    decision = 1.0/(1+np.exp(-np.squeeze(decision)))

                name = str(img_name[0]).replace('/','_')
                sample_names.append(name)

                save_name = name.split('.')[0].split('_',1)[1]
                eval_dir = './output'
                image.tofile('{0}/images/result_{1}.bin'.format(eval_dir, save_name))
                label.tofile('{0}/labels/result_{1}.bin'.format(eval_dir, save_name))

                step += 1

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
























