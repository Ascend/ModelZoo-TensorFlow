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

import argparse
import os


ABSPATH=''
class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_url", type=str, default="")
    parser.add_argument("--train_url", type=str, default="")
    parser.add_argument('--num_gpus', default='')
    # prepro
    parser.add_argument('--vocab_size', default=37000, type=int)#37000

    # train
    ## files
    parser.add_argument('--train1', default=ABSPATH+'iwslt2016/segmented/train.de.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default=ABSPATH+'iwslt2016/segmented/train.en.bpe',
                             help="english training segmented data")
    parser.add_argument('--eval1', default=ABSPATH+'iwslt2016/segmented/eval.de.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default=ABSPATH+'iwslt2016/segmented/eval.en.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default=ABSPATH+'iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary
    parser.add_argument('--vocab', default=ABSPATH+'iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)###128
    parser.add_argument('--eval_batch_size', default=128, type=int)###128

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)####4000
    parser.add_argument('--logdir', default=ABSPATH+"log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)###20
    parser.add_argument('--evaldir', default=ABSPATH+"eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,###512
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,###2048
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,#6
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,###8
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,###100
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,###100
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.1, type=float)#####0.3 论文是0.1
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default=ABSPATH+'iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default=ABSPATH+'iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', default=ABSPATH+"log/1",
    help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)###128
    parser.add_argument('--testdir', default=ABSPATH+"test/1", help="test result dir")
