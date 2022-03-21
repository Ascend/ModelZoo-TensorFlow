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
"""
TRAIN LAUNCHER 

"""
from npu_bridge.npu_init import *

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
flags = tf.flags
FLAGS = flags.FLAGS
#npu parameter
flags.DEFINE_string("precision_mode", "allow_fp32_to_fp16", "Npu Precision Mode")

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('config.cfg')
	
	print('--Creating Dataset')
	dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'], joints= params['joint_list'], modif=False, precision_mode=FLAGS.precision_mode)
	model.generate_model(nEpochs=params['nepochs'])
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)
	
