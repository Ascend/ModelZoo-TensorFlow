from tensorflow.python.tools import freeze_graph


freeze_graph.freeze_graph(
		        input_checkpoint='ckpt/model.ckpt-45000',
		        output_node_names='generator/Sigmoid',
		        output_graph='pb/gan.pb',
                initializer_nodes='',
		        input_graph= None,
		        input_saver= False,
		        input_binary=True,
 		        restore_op_name=None,
		        filename_tensor_name=None,
 		        clear_devices=False,
 		        input_meta_graph='ckpt/model.ckpt-45000.meta')