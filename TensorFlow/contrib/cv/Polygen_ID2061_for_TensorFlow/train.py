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


from npu_bridge.npu_init import *
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt

import modules
import data_utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("datapath", "./meshes", "dataset path")
flags.DEFINE_integer("training_steps", 5000, "training steps")
flags.DEFINE_string("precision_mode", "mix", "precision mode")
flags.DEFINE_string("output_path", "./output", "output path")


def main():
    print("===>>>dataset:{}".format(FLAGS.datapath))
    # Prepare synthetic dataset
    print("===>>>Prepare synthetic dataset")
    ex_list = []

    for k, mesh in enumerate(['cube', 'cylinder', 'cone', 'icosphere']):
        mesh_dict, flag = data_utils.load_process_mesh(
            os.path.join(FLAGS.datapath, '{}.obj'.format(mesh)))
        if flag:
            mesh_dict['class_label'] = k
            ex_list.append(mesh_dict)
    synthetic_dataset = tf.data.Dataset.from_generator(
        lambda: ex_list,
        output_types={
            'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32},
        output_shapes={
            'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]),
            'class_label': tf.TensorShape(())}
    )
    ex = synthetic_dataset.make_one_shot_iterator().get_next()

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./fusion_switch.cfg")
    # custom_op.parameter_map["enable_dump"].b = True
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/npu/ID2061/data_dump")
    # dump_step：指定采集哪些迭代的Dump数据
    # custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
    # dump_mode：Dump模式，取值：input/output/all
    # custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    # import precision_tool.tf_config as npu_tf_config
    # custom_op = npu_tf_config.update_custom_op(custom_op, action='dump')

    # Mixed Precision
    if FLAGS.precision_mode == 'mix':
        print("precision mode: mix")
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    # Inspect the first mesh
    with tf.Session(config=npu_config_proto()) as sess:
        ex_np = sess.run(ex)
    print(ex_np)

    # Plot the meshes
    mesh_list = []
    with tf.Session(config=npu_config_proto()) as sess:
        for i in range(4):
            ex_np = sess.run(ex)
            mesh_list.append(
                {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
                 'faces': data_utils.unflatten_faces(ex_np['faces'])})
    data_utils.plot_meshes(mesh_list, ax_lims=0.4)

    print("===>>>Prepare vertex model")
    # Prepare the dataset for vertex model training
    vertex_model_dataset = data_utils.make_vertex_model_dataset(
        synthetic_dataset, apply_random_shift=False)
    vertex_model_dataset = vertex_model_dataset.repeat()
    vertex_model_dataset = vertex_model_dataset.padded_batch(
        4, padded_shapes=vertex_model_dataset.output_shapes)
    vertex_model_dataset = vertex_model_dataset.prefetch(1)
    vertex_model_batch = vertex_model_dataset.make_one_shot_iterator().get_next()

    # Create vertex model
    vertex_model = modules.VertexModel(
        decoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        class_conditional=True,
        num_classes=4,
        max_num_input_verts=250,
        quantization_bits=8,
    )
    vertex_model_pred_dist = vertex_model(vertex_model_batch)
    vertex_model_loss = -tf.reduce_sum(
        vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) *
        vertex_model_batch['vertices_flat_mask'])
    vertex_samples = vertex_model.sample(
        4, context=vertex_model_batch, max_sample_length=200, top_p=0.95,
        recenter_verts=False, only_return_complete=False)

    print(vertex_model_batch)
    print(vertex_model_pred_dist)
    print(vertex_samples)

    print("===>>>Prepare face model")
    face_model_dataset = data_utils.make_face_model_dataset(
        synthetic_dataset, apply_random_shift=False)
    face_model_dataset = face_model_dataset.repeat()
    face_model_dataset = face_model_dataset.padded_batch(
        4, padded_shapes=face_model_dataset.output_shapes)
    face_model_dataset = face_model_dataset.prefetch(1)
    face_model_batch = face_model_dataset.make_one_shot_iterator().get_next()

    # Create face model
    face_model = modules.FaceModel(
        encoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        decoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        class_conditional=False,
        max_seq_length=500,
        quantization_bits=8,
        decoder_cross_attention=True,
        use_discrete_vertex_embeddings=True,
    )
    face_model_pred_dist = face_model(face_model_batch)
    face_model_loss = -tf.reduce_sum(
        face_model_pred_dist.log_prob(face_model_batch['faces']) *
        face_model_batch['faces_mask'])
    face_samples = face_model.sample(
        context=vertex_samples, max_sample_length=500, top_p=0.95,
        only_return_complete=False)

    print(face_model_batch)
    print(face_model_pred_dist)
    print(face_samples)

    # Optimization settings
    learning_rate = 5e-4
    training_steps = FLAGS.training_steps
    check_step = 5
    plot_step = 100

    # Create an optimizer an minimize the summed log probability of the mesh
    # sequences
    optimizer = tf.train.AdamOptimizer(learning_rate)
    vertex_model_optim_op = optimizer.minimize(vertex_model_loss)
    face_model_optim_op = optimizer.minimize(face_model_loss)

    print("===>>>Training")
    # Training start time
    start_time = time.time()

    # Training loop
    # config = npu_config_proto(config_proto=config_proto)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for n in range(training_steps):
            if n % check_step == 0:
                v_loss, f_loss = sess.run((vertex_model_loss, face_model_loss))
                print('Step {}'.format(n))
                print('Loss (vertices) {}'.format(v_loss))
                print('Loss (faces) {}'.format(f_loss))
                v_samples_np, f_samples_np = sess.run(
                    (vertex_samples, face_samples))
                mesh_list = []
                if n % plot_step == 0:
                    for n in range(4):
                        mesh_list.append(
                            {
                                'vertices': v_samples_np['vertices'][n][:v_samples_np['num_vertices'][n]],
                                'faces': data_utils.unflatten_faces(
                                    f_samples_np['faces'][n][:f_samples_np['num_face_indices'][n]])
                            }
                        )
                    # data_utils.plot_meshes(mesh_list, ax_lims=0.5)
            sess.run((vertex_model_optim_op, face_model_optim_op))
        # Saving model
        # saver = tf.train.Saver()
        # saver.save(sess, os.join(FLAGS.ckpt_path,'model.ckpt'))

    # Training end time
    end_time = time.time()
    print('''TimetoTrain: %4.4f ''' % (end_time - start_time))
    print('''StepTime: %4.4f ''' % ((end_time - start_time) / training_steps))


if __name__ == '__main__':
    main()
