"""
net
"""


# coding=utf-8
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

class Net(object):
    """
    net
    """

    def __init__(self, model):
        """

        Args:
            model:
        """
        self.model = model

    def inference(self, pre_proccessed_data):
        """

        Args:
            pre_proccessed_data:

        Returns:

        """
        return self.model.inference(pre_proccessed_data)

    @staticmethod
    def pre_process(input_data):
        """

        Args:
            input_data:

        Returns:

        """
        return input_data

    @staticmethod
    def post_process(result):
        """

        Args:
            result:

        Returns:

        """
        return result

    def process(self, input_data):
        """

        Args:
            input_data:

        Returns:

        """
        pre_processed_data = Net.pre_process(input_data)
        result = self.inference(pre_processed_data)
        return Net.post_process(result)


if __name__ == '__main__':
    pass
