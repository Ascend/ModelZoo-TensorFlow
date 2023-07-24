import os
import numpy as np
import tensorflow as tf
import amct_tensorflow as amct
import argparse

amct.set_logging_level(print_level="info", save_level="info")


def load_bin(bin_path):
    input_list = []
    bin_path_list = os.listdir(bin_path)
    bin_path_list.sort()
    for idx in range(100):
        input = np.fromfile(os.path.join(bin_path, bin_path_list[idx]), np.float32).reshape(160, 160, 3)
        bin_path.append(input)
    return np.array(input_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to the data directory containing aligned LFE face patches.')
    parser.add_argument('calibration', type=str, help='Data preprocessing output.')
    parser.add_argument('output', type=str, help='Data preprocessing output.')
    args = parser.parse_args()
    calibration_path = os.path.realpath(args.calibration)
    model_path = os.path.realpath(args.model)
    with tf.io.gfile.GFile(model_path, mode='rb') as model:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model.read())
    tf.import_graph_def(graph_def, name='')
    graph = tf.compat.v1.get_default_graph()
    input_tensor = graph.get_tensor_by_name('input' + ':0')
    output_tensor = graph.get_tensor_by_name('embeddings' + ':0')

    calibration_list = os.listdir(calibration_path)
    input_bin = np.fromfile(os.path.join(calibration_path, calibration_list[0]), np.float32).reshape(1, 160, 160, 3)
    with tf.compat.v1.Session() as session:
        origin_prediction = session.run(output_tensor, feed_dict={input_tensor: input_bin})
    config_path = os.path.join(args.output, 'config.json')
    amct.create_quant_config(config_path=config_path, graph=graph, skip_layers=[], batch_num=1)
    record_path = os.path.join(args.output, 'record,txt')
    amct.quantize_model(graph=graph, config_file=config_path, record_path=record_path)
    batch = load_bin(calibration_path)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(output_tensor, feed_dict={input_tensor: batch})
    amct.save_model(
        pb_model=model_path, outputs=['embeddings'], record_file=record_path,
        save_path=os.path.join(args.output, 'facenet')
    )