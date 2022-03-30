import tensorflow as tf


def read_checkpoint():
    w = []
    checkpoint_path = 'dnn_best_model/ckpt_noshuffDIEN3'
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var = reader.get_variable_to_shape_map()
    for key in var:
        # if 'moving' in key:
        # if 'weights' in key and 'conv' in key and 'Mo' not in key:
        print('tensorname:', key)
        print(reader.get_tensor(key))


if __name__ == '__main__':
    # tf.app.run()
    read_checkpoint()