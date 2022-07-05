import tensorflow as tf
from tensorflow.python.tools import freeze_graph


freeze_graph.freeze_graph(
		        input_saved_model_dir='/home/test_user03/tpu-3/models/official/amoeba_net/1656992029',
		        output_node_names='predictions',
		        output_graph='test_299.pb',
                        initializer_nodes='',
		        input_graph= None,
		        input_saver= False,
		        input_binary=False, 
		        input_checkpoint=None, 
 		        restore_op_name=None,
		        filename_tensor_name=None,
 		        clear_devices=False,
 		        input_meta_graph=False)