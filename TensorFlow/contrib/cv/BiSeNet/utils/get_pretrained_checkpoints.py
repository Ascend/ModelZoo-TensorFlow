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
# ==============================================================================
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()


if args.model == "ResNet50" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz', "-P", "pretrain"])
	try:
		subprocess.check_output(['tar', '-xvf', 'pretrain/resnet_v2_50_2017_04_14.tar.gz', "-C", "pretrain"])
		subprocess.check_output(['rm', 'pretrain/resnet_v2_50_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "ResNet101" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz', "-P", "pretrain"])
	try:
		subprocess.check_output(['tar', '-xvf', 'pretrain/resnet_v2_101_2017_04_14.tar.gz', "-C", "pretrain"])
		subprocess.check_output(['rm', 'pretrain/resnet_v2_101_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "ResNet152" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz', "-P", "pretrain"])
	try:
		subprocess.check_output(['tar', '-xvf', 'pretrain/resnet_v2_152_2017_04_14.tar.gz', "-C", "pretrain"])
		subprocess.check_output(['rm', 'pretrain/resnet_v2_152_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "MobileNetV2" or args.model == "ALL":
	subprocess.check_output(['wget','https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz', "-P", "pretrain"])
	try:
		subprocess.check_output(['tar', '-xvf', 'pretrain/mobilenet_v2_1.4_224.tgz', "-C", "pretrain"])
		subprocess.check_output(['rm', 'pretrain/mobilenet_v2_1.4_224.tgz'])
	except Exception as e:
		print(e)
		pass

if args.model == "InceptionV4" or args.model == "ALL":
	subprocess.check_output(
		['wget', 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz', "-P", "pretrain"])
	try:
		subprocess.check_output(['tar', '-xvf', 'pretrain/inception_v4_2016_09_09.tar.gz', "-C", "pretrain"])
		subprocess.check_output(['rm', 'pretrain/inception_v4_2016_09_09.tar.gz'])
	except Exception as e:
		print(e)
		pass
