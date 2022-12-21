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

import xlrd
from openpyxl import load_workbook
import os
import json
from collections import OrderedDict


def read_xlrd(excel_file):
    """
    读取excel文件
    """
    data = xlrd.open_workbook(excel_file)
    table = data.sheet_by_index(0)
    # print(table.row_values[0])
    head = [i_name for i_name in table.row_values(0)]
    return head, table


def add_excel_sheet(dataframe, excel_writer, sheet_name):
    """
    往excel添加sheet
    """
    book = load_workbook(excel_writer.path)
    excel_writer.book = book
    dataframe.to_excel(excel_writer=excel_writer, sheet_name=sheet_name)
    excel_writer.close()


def read_json_format_file(file):
    """
    读取每行为json格式的文本
    :param file: 文件名
    :return: 每行文本
    """
    if not os.path.exists(file):
        raise FileNotFoundError("【{}】文件未找到，请检查".format(file))
    print(">>>>> 正在读原始取数据文件：{}".format(file))
    line_count = 0
    with open(file, 'r') as f:
        while True:
            _line = f.readline()
            line_count += 1
            if not _line:
                break
            else:
                line = json.loads(_line.strip())
                if line_count % 100000 == 0:
                    print(">>>>> 已读取{}行".format(line_count))
                yield line


def dict_sort(result, limit_num=None):
    """
    字典排序, 返回有序字典
    :param result:
    :param limit_num:
    :return:
    """
    _result_sort = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result_sort = OrderedDict()

    count_limit = 0
    domain_count = 0
    for i in _result_sort:
        if limit_num:
            if i[1] > limit_num:
                result_sort[i[0]] = i[1]
                domain_count += 1
                count_limit += i[1]
        else:
            result_sort[i[0]] = i[1]
    return result_sort