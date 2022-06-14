import tensorflow as tf
from google.protobuf import text_format
def convert_pb_to_pbtxt():
    """
    :param filename:
    :return:
    """
    with tf.gfile.GFile('./npuzh.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', 'npuzh.pbtxt', as_text=True)
    return
def convert_pbtxt_to_pb():
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.GFile('./npuzh.pbtxt') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', 'npuzh.pb', as_text=False)
    return
if __name__ == '__main__':   
    convert_pb_to_pbtxt()
