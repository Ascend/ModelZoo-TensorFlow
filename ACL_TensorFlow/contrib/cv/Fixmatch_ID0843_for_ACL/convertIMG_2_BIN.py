# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
import os
import tensorflow as tf
from absl import flags,app
import numpy as np
from fixmatch import FixMatch
from libml import data,utils
FLAGS = flags.FLAGS
def main(argv):
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, FixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    input_folder = os.path.exists("./input_bin_{0:02d}".format(FLAGS.batchsize))
    output_folder = os.path.exists("./output_label_{0:02d}".format(FLAGS.batchsize)) 
    if not input_folder:
        os.makedirs("./input_bin_{0:02d}".format(FLAGS.batchsize))
    if not output_folder:
        os.makedirs("./output_label_{0:02d}".format(FLAGS.batchsize))
    with tf.Session(config=utils.get_config()) as sess:
        model.session = sess
        model.cache_eval()
        images, labels = model.tmp.cache['test']
        for j in range(0,len(labels),FLAGS.batchsize):
            x = images[j:j+FLAGS.batchsize]
            label = labels[j:j+FLAGS.batchsize]
            np.array(x).tofile("./input_bin_{0:02d}/{1:05d}.bin".format(FLAGS.batchsize,j))
            np.savetxt("./output_label_{0:02d}/{1:05d}.txt".format(FLAGS.batchsize,j),label)


if __name__ == '__main__':
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_integer('batchsize',1,'the batch number of IMG in one bin file')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.5@40-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)



    app.run(main)
    # python convertIMG2BIN.py --batchsize=1 --dataset=cifar10.5@40-1