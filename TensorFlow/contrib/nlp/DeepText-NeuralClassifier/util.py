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

import logging
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


LABEL_SEPARATOR = "--"
CHARSET = "utf-8"
EPS = 1e-7


class Logger(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if config.log.log_level == "debug":
            logging_level = logging.DEBUG
            tf_logging_level = tf.compat.v1.logging.DEBUG
        elif config.log.log_level == "info":
            logging_level = logging.INFO
            tf_logging_level = tf.compat.v1.logging.INFO
        elif config.log.log_level == "warn":
            logging_level = logging.WARN
            tf_logging_level = tf.compat.v1.logging.WARN
        elif config.log.log_level == "error":
            logging_level = logging.ERROR
            tf_logging_level = tf.compat.v1.logging.ERROR
        else:
            raise TypeError(
                "No logging type named %s, candidate is: info, debug, error")
        format_str = '[%(asctime)s] [%(levelname)s] [%(filename)s-%(lineno)d] [%(message)s]'
        logging.basicConfig(filename=config.log.logger_file,
                            level=logging_level,
                            format=format_str,
                            filemode="a", datefmt='%Y-%m-%d %H:%M:%S')
        tf.compat.v1.logging.set_verbosity(tf_logging_level)

    @staticmethod
    def debug(msg):
        """Log debug message
            msg: Message to log
        """
        tf.compat.v1.logging.debug(msg)

    @staticmethod
    def info(msg):
        """"Log info message
            msg: Message to log
        """
        tf.compat.v1.logging.info(msg)

    @staticmethod
    def warn(msg):
        """Log warn message
            msg: Message to log
        """
        tf.compat.v1.logging.warn(msg)

    @staticmethod
    def error(msg):
        """Log error message
            msg: Message to log
        """
        tf.compat.v1.logging.error(msg)

    @staticmethod
    def flush():
        """Log flush message
            msg: Message to log
        """
        tf.compat.v1.logging.flush()


def main():
    from config import Config
    config_file = "conf/fasttext_token_char.config"
    config = Config(config_file=config_file)

    logger = Logger(config)
    logger.debug("test debug msg")
    logger.info("test info msg")
    logger.warn("test warning msg")
    logger.error("test error msg")


if __name__ == '__main__':
    main()