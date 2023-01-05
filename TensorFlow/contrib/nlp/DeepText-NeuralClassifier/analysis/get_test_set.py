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

import json
from general_tools.utils import read_xlrd, read_json_format_file


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

    return ",".join([t.split("|")[0].replace("'", "") for t in tag_list])


def process_manual_tonality(tag, manual_tag):
    """
    处理tonality调性分类
    """
    if str(tag).lower() in ["0.0", "false"]:
        tag = "false"
    elif str(tag).lower() in ["1.0", "true"]:
        tag = "true"
    else:
        tag = tag

    if str(manual_tag) in ["1", "1.0"]:
        standard_tag = tag
    else:
        if str(manual_tag).lower() in ["0.0", "false"]:
            standard_tag = "false"
        else:
            standard_tag = manual_tag

    return standard_tag


def write_validation_set_to_file(excel_file, outfile):
    """
    将点检信息处理为文本
    """
    head, table = read_xlrd(excel_file)
    # 一二级分类
    alg_tag2_index = head.index("alg_tag-2.0")
    manual_tag2_index = head.index("manual_tag-2.0")
    alg_subtag2_index = head.index("alg_subtag-2.0")
    manual_subtag2_index = head.index("manual_subtag-2.0")
    alg_tag3_index = head.index("alg_tag-3.0")
    manual_tag3_index = head.index("manual_tag-3.0")
    alg_subtag3_index = head.index("alg_subtag-3.0")
    manual_subtag3_index = head.index("manual_subtag-3.0")
    # 低调性分类
    alg_vulgar_index = head.index("alg_vulgar")
    manual_vulgar_index = head.index("manual_vulgar")
    alg_gossip_index = head.index("alg_gossip")
    manual_gossip_index = head.index("manual_gossip")
    alg_clickbait_index = head.index("alg_clickbait")
    manual_clickbait_index = head.index("manual_clickbait")
    alg_advert_index = head.index("alg_advert")
    manual_advert_index = head.index("manual_advert")
    file = open(outfile, "w", encoding="utf-8")
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        # print(type(row_value[alg_tag2_index]))
        row_value[manual_tag2_index] = process_manual_category(row_value[alg_tag2_index], row_value[manual_tag2_index])
        row_value[manual_subtag2_index] = process_manual_category(row_value[alg_subtag2_index], row_value[manual_subtag2_index])
        row_value[manual_tag3_index] = process_manual_category(row_value[alg_tag3_index], row_value[manual_tag3_index])
        row_value[manual_subtag3_index] = process_manual_category(row_value[alg_subtag3_index], row_value[manual_subtag3_index])

        row_value[manual_vulgar_index] = process_manual_tonality(row_value[alg_vulgar_index], row_value[manual_vulgar_index])
        row_value[manual_gossip_index] = process_manual_tonality(row_value[alg_gossip_index], row_value[manual_gossip_index])
        row_value[manual_clickbait_index] = process_manual_tonality(row_value[alg_clickbait_index], row_value[manual_clickbait_index])
        row_value[manual_advert_index] = process_manual_tonality(row_value[alg_advert_index], row_value[manual_advert_index])

        row_line = dict(zip(head, row_value))
        file.write(json.dumps(row_line, ensure_ascii=False) + "\n")

    file.close()

def _label_add(key, label_dict):
    if key in label_dict:
        label_dict[key] += 1
    else:
        label_dict[key] = 1


def label_count(file):
    all_label = dict()
    all_label["tag2.0"] = {"top": {"one_label": 0, "multi_label": 0}, "sub": {"one_label": 0, "multi_label": 0}}
    all_label["tag3.0"] = {"top": {"one_label": 0, "multi_label": 0}, "sub": {"one_label": 0, "multi_label": 0}}
    for line in read_json_format_file(file):
        top_category2 = line["manual_tag-2.0"]
        if top_category2 == "0.0":
            print(line)
        sub_category2 = line["manual_subtag-2.0"]
        top_category3 = line["manual_tag-3.0"]
        sub_category3 = line["manual_subtag-3.0"]
        if "," not in top_category2 and top_category2 != "":
            all_label["tag2.0"]["top"]["one_label"] += 1
            # label = top_category2.split("_")[0]

        else:
            all_label["tag2.0"]["top"]["multi_label"] += 1
        _label_add(top_category2, all_label["tag2.0"]["top"])

        if "," not in sub_category2 and sub_category2 != "":
            all_label["tag2.0"]["sub"]["one_label"] += 1
        else:
            all_label["tag2.0"]["sub"]["multi_label"] += 1
        _label_add(sub_category2, all_label["tag2.0"]["sub"])

        if "," not in top_category3 and top_category3 != "":
            all_label["tag3.0"]["top"]["one_label"] += 1
        elif top_category3 == "":
            continue
        else:
            all_label["tag3.0"]["top"]["multi_label"] += 1
        _label_add(top_category3, all_label["tag3.0"]["top"])

        if "," not in sub_category3 and sub_category3 != "":
            all_label["tag3.0"]["sub"]["one_label"] += 1
        elif sub_category3 == "":
            continue
        else:
            all_label["tag3.0"]["sub"]["multi_label"] += 1
        _label_add(sub_category3, all_label["tag3.0"]["sub"])

    print(json.dumps(all_label, ensure_ascii=False, indent=4))
    return all_label


def write_validation3_set_to_file(file, outfile):
    """
    单独写标签3.0体系到文件
    """
    tag3 = ["社会", "体育", "娱乐", "财经", "时政", "科技", "时尚", "教育", "情感", "文化",
            "旅游", "美食", "宠物", "星座命理", "搞笑", "壁纸头像", "生活", "职场", "小说",
            "国际", "房产", "汽车", "军事", "游戏", "动漫", "育儿", "健康", "历史", "儿童",
             "知识", "其他"]

    with open(file, "r", encoding="utf-8") as f, open(outfile, "w", encoding="utf-8") as wf:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line.strip())
            label = line_json["manual_tag-3.0"]
            if label == "":
                continue
            else:
                in_lable = True
                if "," in label:
                    label_list = label.split(",")
                    for lab in label_list:
                        if lab not in tag3:
                            in_lable = False
                else:
                    if label not in tag3:
                        in_lable = False
                if in_lable:
                    wf.write(line)



def main():
    excel_file = "/work/data/9-25周周点检数据/all_spot_check.xlsx"
    outfile = "./all_tag_test.txt"
    validation_set_outfile = "./tag3.0_test_corpus.txt"
    # write_validation_set_to_file(excel_file, outfile)
    write_validation3_set_to_file(outfile, validation_set_outfile)
    label_count(validation_set_outfile)


if __name__ == "__main__":
    main()


