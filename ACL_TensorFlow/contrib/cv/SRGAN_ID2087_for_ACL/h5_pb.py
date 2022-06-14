import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2 
def h5_to_pb(h5_save_path):
    model = tf.keras.models.load_model(h5_save_path, compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models3",
                      name="npuzh.pb",
                      as_text=False)
h5_to_pb('./gen_model500.h5')
