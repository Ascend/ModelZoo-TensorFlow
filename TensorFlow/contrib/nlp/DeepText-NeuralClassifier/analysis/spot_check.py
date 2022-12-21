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
from sklearn import metrics
from general_tools.utils import read_xlrd, add_excel_sheet


def get_category_result(head, table):
    """
    获取标注信息，人工标注，ai标注，算法标注
    """
    ai_tag_result = list()
    ai_subtag_result = list()
    alg_tag2_result = list()
    alg_subtag2_result = list()
    alg_tag3_result = list()
    alg_subtag3_result = list()
    manual_tag2_result = list()
    manual_subtag2_result = list()
    manual_tag3_result = list()
    manual_subtag3_result = list()

    ai_tag_index = head.index("ai_tag")
    ai_subtag_index = head.index("ai_subtag")
    alg_tag2_index = head.index("alg_tag-2.0")
    alg_subtag2_index = head.index("alg_subtag-2.0")
    alg_tag3_index = head.index("alg_tag-3.0")
    alg_subtag3_index = head.index("alg_subtag-3.0")
    manual_tag2_index = head.index("manual_tag-2.0")
    manual_subtag2_index = head.index("manual_subtag-2.0")
    manual_tag3_index = head.index("manual_tag-3.0")
    manual_subtag3_index = head.index("manual_subtag-3.0")
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        # print(type(row_value))
        ai_tag_result.append(get_category(row_value[ai_tag_index]))
        ai_subtag_result.append(get_category(row_value[ai_subtag_index]))
        alg_tag2_result.append(get_category(row_value[alg_tag2_index]))
        alg_subtag2_result.append(get_category(row_value[alg_subtag2_index]))
        alg_tag3_result.append(get_category(row_value[alg_tag3_index]))
        alg_subtag3_result.append(get_category(row_value[alg_subtag3_index]))
        manual_tag2_result.append(get_category(row_value[manual_tag2_index]))
        manual_subtag2_result.append(get_category(row_value[manual_subtag2_index]))
        manual_tag3_result.append(process_manual_category(row_value[alg_tag3_index], row_value[manual_tag3_index]))
        manual_subtag3_result.append(process_manual_category(row_value[alg_subtag3_index], row_value[manual_subtag3_index]))

    return (ai_tag_result, ai_subtag_result), \
           (alg_tag2_result, alg_subtag2_result), \
           (alg_tag3_result, alg_subtag3_result), \
           (manual_tag2_result, manual_subtag2_result),\
           (manual_tag3_result, manual_subtag3_result)


def get_category(tag):
    """
    处理category信息
    """
    if "[" and "]" not in tag:
        tag_list = tag.split(",")
    else:
        tag_list = eval(tag)

    return "\t".join([t.split("|")[0] for t in tag_list])

def process_manual_category(tag, manual_tag):
    """
    处理category信息
    """
    replace_tag = str(manual_tag)

    if str(manual_tag) in ["1", "1.0"]:
        replace_tag = tag

    if "[" and "]" not in replace_tag:
        tag_list = replace_tag.split(",")
    else:
        tag_list = eval(replace_tag)

    return "\t".join([t.split("|")[0].replace("'", "") for t in tag_list])



def calculate(file):
    """
    计算指标，并写入excel文件
    """
    head, table = read_xlrd(file)
    (ai_tag_result, ai_subtag_result), \
    (alg_tag2_result, alg_subtag2_result), \
    (alg_tag3_result, alg_subtag3_result), \
    (manual_tag2_result, manual_subtag2_result), \
    (manual_tag3_result, manual_subtag3_result) = get_category_result(head, table)
    # print(ai_tag_result)
    # print(ai_subtag_result)

    file_path = "./spot_check_result_26.xlsx"
    # 创建一个空的excel文件
    nan_excel = pd.DataFrame()
    nan_excel.to_excel(file_path)
    writer = pd.ExcelWriter(file_path, engine="openpyxl")

    ai_tag_df, ai_precision = get_df_result(manual_tag2_result, ai_tag_result)
    alg_tag2_df, alg_tag2_precision = get_df_result(manual_tag2_result, alg_tag2_result)
    alg_tag3_df, alg_tag3_precision = get_df_result(manual_tag3_result, alg_tag3_result)
    ai_subtag_df, ai_subtag_precision = get_df_result(manual_subtag2_result, ai_subtag_result)
    alg_subtag2_df, alg_subtag2_precision = get_df_result(manual_subtag2_result, alg_subtag2_result)
    alg_subtag3_df, alg_subtag3_precision = get_df_result(manual_subtag3_result, alg_subtag3_result)
    # precision_list = dict(zip(report_df.index, report_df.precision))
    # confusion_matrix_top = metrics.confusion_matrix(y_true=manual_tag_result, y_pred=ai_tag_result)
    # print(json.dumps(precision_list, indent=4, ensure_ascii=False))
    # print(confusion_matrix_top)
    # 分类准确率
    precision = dict()
    precision["一级分类"] = {"ai-tag": ai_precision, "alg-tag-2.0": alg_tag2_precision, "alg-tag-3.0": alg_tag3_precision}
    precision["二级分类"] = {"ai-tag": ai_subtag_precision, "alg-tag-2.0": alg_subtag2_precision,
                         "alg-tag-3.0": alg_subtag3_precision}
    precision_df = pd.DataFrame(precision).T
    add_excel_sheet(precision_df, writer, "precision")
    # 一级分类classification report
    add_excel_sheet(ai_tag_df, writer, "ai-tag")
    add_excel_sheet(alg_tag2_df, writer, "alg-tag-2.0")
    add_excel_sheet(alg_tag3_df, writer, "alg-tag-3.0")
    # 二级分类classification report
    add_excel_sheet(ai_subtag_df, writer, "ai-subtag")
    add_excel_sheet(alg_subtag2_df, writer, "alg-subtag-2.0")
    add_excel_sheet(alg_subtag3_df, writer, "alg-subtag-3.0")



def get_df_result(y_true, y_pre):
    """
    写结果信息到文件
    """
    report_top = metrics.classification_report(y_true=y_true, y_pred=y_pre, output_dict=True)
    precision = metrics.accuracy_score(y_true=y_true, y_pred=y_pre)

    # new_result = {}
    # for k, v in report_top.items():
    #     if v["support"] == 0.0:
    #         continue
    #     new_result[k] = v
    report_df = pd.DataFrame(report_top).T
    # support改为int型
    report_df['support'] = report_df['support'].astype(int)
    # 按support排序，不排序average部分
    # report_df = report_df.loc[report_df["support"] != 0]
    report_df = report_df.iloc[:-3].sort_values(by=['support'], ascending=False)
    # print(report_df)
    return report_df, precision



def main():
    file = "/work/data/category/点检数据/内容质量点检2020-26周.xlsx"
    calculate(file)


if __name__ == "__main__":
    main()