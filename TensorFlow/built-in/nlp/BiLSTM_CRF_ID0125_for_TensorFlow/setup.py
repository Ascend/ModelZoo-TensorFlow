# encoding =utf-8
#
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
#

from npu_bridge.npu_init import *
from os import path
import codecs
from setuptools import setup, find_packages

# setup metainfo
# libinfo_py = 'bert_lstm_ner.py'
# libinfo_content = open(libinfo_py, 'r', encoding='utf-8').readlines()
# version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
# # exec(version_line)  # produce __version__
# __version__ = version_line.split('=')[1].replace(' ', '')
# print(__version__)
setup(
    name='bert_base',
    version='0.0.9',
    description='Use Google\'s BERT for Chinese natural language processing tasks such as named entity recognition and provide server services',
    url='https://github.com/macanv/BERT-BiLSTM-CRF-NER',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Ma Can',
    author_email='ma_cancan@163.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'six',
        'pyzmq>=16.0.0',
        'GPUtil>=1.3.0',
        'termcolor>=1.1',
    ],
    extras_require={
        'cpu': ['tensorflow>=1.10.0'],
        'gpu': ['tensorflow-gpu>=1.10.0'],
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json']
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        #'Topic :: Scientific/Engineering :: Artificial Intelligence :: Natural Language Processing :: Named Entity Recognition',
    ),
    entry_points={
        'console_scripts': ['bert-base-serving-start=bert_base.runs:start_server',
                            'bert-base-ner-train=bert_base.runs:train_ner'],
    },
    keywords='bert nlp ner NER named entity recognition bilstm crf tensorflow machine learning sentence encoding embedding serving',
)

