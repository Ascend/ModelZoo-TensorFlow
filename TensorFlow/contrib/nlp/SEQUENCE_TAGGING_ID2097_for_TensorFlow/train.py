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

import os
import time

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
print('ASCEND_GLOBAL_LOG_LEVEL = ', os.environ.get('ASCEND_GLOBAL_LOG_LEVEL'))


def main():
    # create instance of config
    config = Config(load=False)

    start_time = time.time()

    # load vocab
    config.load()

    # build model
    model = NERModel(config)
    model.build()

    # create datasets
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, test)

    # close session
    model.close_session()

    end_time = time.time()
    print('Finish training! time cost: ', (end_time - start_time) / 3600, 'hours')


if __name__ == "__main__":
    main()
