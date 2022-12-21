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

import pandas as pd
import time
# from pandas.io.json import json_normalize
from sklearn import metrics
from collections import Counter

from general_tools.utils import read_json_format_file, read_xlrd


def write_kafka_data_to_excel(file, excel_file):
    """
    把点检数据写入excel
    """
    out = []

    for line in read_json_format_file(file):
        out.append(line)
    df = pd.DataFrame(out)
    # print(df)
    df.to_excel(excel_file, index=False)


def calculate_index(file, topcategory_pr_file):
    """
    计算机器打标的微平均准确率和召回率
    """
    head, table = read_xlrd(file)
    alg_top_label_index = head.index("algFirstCategory")
    alg_sub_label_index = head.index("algSecondCategory")
    diting_top_label_index = head.index("ditingFirstCategory")
    diting_sub_label_index = head.index("ditingSecondCategory")

    top_label_true = list()
    top_label_pred = list()
    sub_label_true = list()
    sub_label_pred = list()
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        top_label_pred.append(get_standard_label(row_value[alg_top_label_index]))
        top_label_true.append(get_standard_label(row_value[diting_top_label_index]))
        sub_label_pred.append(get_standard_label(row_value[alg_sub_label_index]))
        sub_label_true.append(get_standard_label(row_value[diting_sub_label_index]))
    top_report = get_sorted_report(top_label_true, top_label_pred)
    # print(top_report)
    sub_report = get_sorted_report(sub_label_true, sub_label_pred)
    # print(sub_report)

    true_label_count = Counter([la for lable in top_label_true for la in lable.split(",") if la != ""])
    print(true_label_count)
    calculate_topcategory_pr(topcategory_pr_file, true_label_count)


def get_sorted_report(y_true, y_pred):
    """
    获取分类报告
    """
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df['support'] = report_df['support'].astype(int)
    # 按support排序，不排序average部分
    # report_df = report_df.loc[report_df["support"] != 0]
    # report_df = report_df.iloc[:-3].sort_values(by=['support'], ascending=False)

    return report_df


def get_standard_label(tag):
    """
    处理标注结果为标准标签
    """
    if "[" and "]" not in tag:
        tag_list = tag.split(",")
    else:
        tag_list = eval(tag)

    return ",".join([t.split("|")[0] for t in tag_list])


def calculate_topcategory_pr(file, true_label_count):

    head, table = read_xlrd(file)
    topcategory_dict = dict()
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        name = row_value[head.index("name")]
        p = float(row_value[head.index("p")])
        r = float(row_value[head.index('r')])
        topcategory_dict[name] = (p, r)
    print(_calculate_micro_precision(topcategory_dict, true_label_count))
    print(_calculate_micro_recall(topcategory_dict, true_label_count))


def _calculate_micro_precision(topcategory_dict, true_label_count):
    pp = 0
    tt = 0
    for item in topcategory_dict:
        spt = true_label_count[item]
        tt += spt * topcategory_dict[item][1]
        pp += spt * topcategory_dict[item][1] / topcategory_dict[item][0]
    return tt / pp


def _calculate_micro_recall(topcategory_dict, true_label_count):

    tt = 0
    for item in topcategory_dict:
        spt = true_label_count[item]
        tt += spt * topcategory_dict[item][1]
    return tt / sum([true_label_count[item] for item in topcategory_dict])


def main():

    now_time = time.strftime("%Y-%m-%d", time.localtime())
    file = "data_kafka"
    excel_file = "diting_spot_check_2020-07-01.xlsx"
    topcategory_pr_file = "calculate_topcategory_pr.xlsx"
    # write_kafka_data_to_excel(file, excel_file)
    calculate_index(excel_file, topcategory_pr_file)


if __name__ == "__main__":
    main()