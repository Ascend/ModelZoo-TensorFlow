"""
log
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

import logging
import gman_flags as df


def def_logger(path):
    """

    Args:
        path:

    Returns:

    """
    LOGGING_LEVEL = logging.DEBUG

    logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s  [ %(levelname)s ]: %(message)s')
    logger = logging.getLogger(__name__)

    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    def info(message):
        """

        Args:
            message:

        Returns:

        """
        logger.info(message)

    def warn(message):
        """

        Args:
            message:

        Returns:

        """
        logger.warning(message)

    def error(message):
        """

        Args:
            message:

        Returns:

        """
        logger.error(message)

    def debug(message):
        """

        Args:
            message:

        Returns:

        """
        logger.debug(message)

    return logger


if __name__ == '__main__':
    pass
