# coding=utf-8
import os
import re
import time
import sys
from lib.util import util
from lib.constant import Constant
from lib.h5_util import H5Util
import config as cfg
from lib.precision_tool_exception import catch_tool_exception
from lib.precision_tool_exception import PrecisionToolException


class PtDump(object):
    def __init__(self, data_dir):
        self.log = util.get_log()
        self.npu = None
        self.gpu = None
        self.data_dir = data_dir

    def prepare(self):
        util.create_dir(cfg.PT_NPU_DIR)
        util.create_dir(cfg.PT_GPU_DIR)
        if not util.empty_dir(cfg.PT_NPU_DIR):
            self.npu = H5Util()
        if not util.empty_dir(cfg.PT_GPU_DIR):
            self.gpu = H5Util()

    def get_dump_files_by_name(self, file_name):
        """Get dump files by name"""
        print(file_name)
        return {}
