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

import os
import json
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from general_tools.text_preprocess import CleanDoc
from general_tools.utils import read_json_format_file

class PreCorpus(object):
    """
    处理为模型训练数据，包括数据清洗和数据分层
    """
    def __init__(self, corpus_file, out_dir):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        self.clean_doc_obj = CleanDoc()
        self.corpus_file = corpus_file
        self.train_file = os.path.join(out_dir, "train.txt")
        self.dev_file = os.path.join(out_dir, "validate.txt")
        self.test_file = os.path.join(out_dir, "test.txt")
        # self.analysis_label_dist()
        self.get_category_corpus_file()

    def analysis_label_dist(self):
        """
        统计label分布
        """
        category_count = dict()
        for line in read_json_format_file(self.corpus_file):
            _, (top_category, _, _) = self._preline(line)
            self._label_dist_count(level1=top_category, dist_count=category_count)

        sorted_category_count = self.dict_sort(category_count)
        print(json.dumps(sorted_category_count, ensure_ascii=False, indent=4))

    def get_category_corpus_file(self):
        print(">>>>> preprocess corpus")
        X, Y = self.get_clean_data(category_level=1)
        train, dev = self.stratified_sampling(X, Y, 0.2)
        self.write_txt_file(train, self.train_file)
        self.write_txt_file(dev, self.dev_file)
        self.write_txt_file(dev, self.test_file)

    def get_clean_data(self, category_level=1):
        X = list()
        Y = list()
        _count = dict()
        for line in read_json_format_file(self.corpus_file):
            if line:
                (_id, channel, title, content), y = self._preline(line)
                label = y[category_level - 1]
                _, title_char_feature, title_token_feature = self.clean_doc_obj.get_char_and_token_feature(title)
                _, content_char_feature, content_token_feature = self.clean_doc_obj.get_char_and_token_feature(content)
                # 获取title和content的char特征
                char_feature = " ".join(list(title_char_feature)) + " ### " + " ".join(list(content_char_feature))
                # 获取title和content的token特征
                token_feature = " ".join(list(title_token_feature)) + " ### " + " ".join(list(content_token_feature))

                if len(content) > 30:
                    if label in _count.keys():
                        if _count[label] > 30000:
                            continue
                        else:
                            _count[label] += 1
                            X.append(token_feature + "\t" + char_feature + "\t" + channel)
                            Y.append(label)
                    else:
                        _count[label] = 1
                        X.append(token_feature + "\t" + char_feature + "\t" + channel)
                        Y.append(label)
                else:
                    continue
        return X, Y

    def stratified_sampling(self, x, y, valid_portion):
        """
        按标签类别个数分层切分训练集和验证集
        :param x:
        :param y:
        :param valid_portion:
        :return:
        """
        skf = StratifiedKFold(n_splits=int(1 / valid_portion))
        train = None
        dev = None

        index = [(train_index, test_index) for train_index, test_index in skf.split(x, y)]

        train_label_count = self._label_count([y[i] for i in index[0][0]])
        test_label_count = self._label_count([y[j] for j in index[0][1]])
        print("train_label_count: {}".format(json.dumps(train_label_count, indent=4, ensure_ascii=False)))
        print("test_label_count: {}".format(json.dumps(test_label_count, indent=4, ensure_ascii=False)))
        train = [y[i] + "\t" + x[i] for i in index[0][0]]
        dev = [y[j] + "\t" + x[j] for j in index[0][1]]

        return train, dev

    def _label_dist_count(self, level1=None, level2=None, level3=None, dist_count=None):
        """
        统计标签分布计算
        :param level1:一级标签
        :param level2:二级标签
        :param level3:三级标签
        :param dist_count:标签分布字典
        :return:
        """
        if level1:
            if level1 in dist_count:
                dist_count[level1]["count"] += 1
            else:
                dist_count[level1] = dict()
                dist_count[level1]["count"] = 1
            if level2:
                if level2 in dist_count[level1]:
                    dist_count[level1][level2]["count"] += 1
                else:
                    dist_count[level1][level2] = dict()
                    dist_count[level1][level2]["count"] = 1
                if level3:
                    if level3 in dist_count[level1][level2]:
                        dist_count[level1][level2][level3] += 1
                    else:
                        dist_count[level1][level2][level3] = 1

    @staticmethod
    def dict_sort(result, limit_num=None):
        """
        字典排序, 返回有序字典
        :param result:
        :param limit_num:
        :return:
        """
        _result_sort = sorted(result.items(), key=lambda x: x[1]["count"], reverse=True)
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

    @staticmethod
    def _preline(line):
        """
        处理文件行（dict格式）
        """
        article_id = line.get("articleid", "")
        channel = line.get("channel", "")
        title = line.get("title", "")
        content = line.get("content", "")
        tagging_data = eval(line.get("tagscoredata", ""))
        category_data = tagging_data["categoryList"][0]["data"]
        top_category = ""
        sub_category = ""
        third_category = ""
        for item in category_data:
            if item["level"] == 1:
                top_category = str(item["name"])
            if item["level"] == 2:
                if "name" in item:
                    sub_category = str(item["name"].split("_")[-1])
            if item["level"] == 3:
                if "name" in item:
                    third_category = str(item["name"].split("_")[-1])
        return (article_id, channel, title, content), (top_category, sub_category, third_category)

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def write_txt_file(self, data, file):
        """
        写数据到文件
        """
        print(">>>>> start writing file")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< write down：{}".format(file))


class PreCorpusV1(object):
    """
    根据历史数据版本@刘欢
    进行新的数据清洗过程
    """
    def __init__(self, file, outfile):
        self.file = file
        self.outfile = outfile
        self.clean_doc_obj = CleanDoc(stopwords_file="./data/stopwords.txt")
        file_path, file_extend = os.path.splitext(outfile)
        self.corpus_file = "{}_corpus{}".format(file_path, file_extend)
        self.error_file = "{}_error{}".format(file_path, file_extend)
        self.get_clean_data()

    def get_clean_data(self):
        clean_data = []
        with open(self.file, "r", encoding="utf-8") as f, \
                open(self.error_file, "w", encoding="utf-8") as ef, \
                open(self.corpus_file, "w", encoding="utf-8") as cf:
            lines = f.readlines()
            for line in lines:
                line_str = line.strip().split('\t')
                corpus_line = dict()
                if len(line_str) == 4:
                    label = line_str[0]
                    if "#   #   #" in line_str[1]:
                        text = line_str[1].split("#   #   #")
                    else:
                        text = line_str[1].split("###")
                    if len(text) == 2:
                        _title, _content = text[0], text[1]
                    else:
                        _title, _content = "", text[0]
                    title = "".join(_title.split(" "))
                    content = "".join(_content.split(" "))
                    _, title_char_feature, title_token_feature = self.clean_doc_obj.get_char_and_token_feature(title)
                    _, content_char_feature, content_token_feature = self.clean_doc_obj.get_char_and_token_feature(content)

                    # 获取title和content的char特征
                    char_feature = " ".join(list(title_char_feature)) + " ### " + " ".join(
                        list(content_char_feature))
                    # 获取title和content的token特征
                    token_feature = " ".join(list(title_token_feature)) + " ### " + " ".join(
                        list(content_token_feature))
                    clean_data.append(label + "\t" + token_feature + "\t" + char_feature + "\t" + line_str[3])
                    corpus_line["title"] = title
                    corpus_line["content"] = content
                    corpus_line["topcategory"] = label
                    cf.write(json.dumps(corpus_line, ensure_ascii=False) + "\n")
                else:
                    ef.write(line.strip() + "\n")
        self.rewrite_txt_file(clean_data, self.outfile)

    def rewrite_txt_file(self, data, file):
        """
        写数据到文件
        """
        print(">>>>> start writing file")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< write down：{}".format(file))


class PreLargeLabelCorpus(object):
    """
    单独训练大类（社会、时政、国际、军事）
    """
    def __init__(self, corpus_file, outfile):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        self.corpus_file = corpus_file
        self.outfile = outfile
        self.write_select_label_to_file()

    def write_select_label_to_file(self):
        with open(self.corpus_file, "r", encoding="utf-8") as f, \
            open(self.outfile, "w", encoding="utf-8") as wf:
            for line in f.readlines():
                if line.strip().split("\t")[0] in ["社会", "国际", "时政", "军事"]:
                    wf.write(line)


class PreTestCorpus(object):
    """
    单独处理点检数据作为测试集
    """

    def __init__(self, corpus_file, outfile):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        self.clean_doc_obj = CleanDoc(stopwords_file="./data/stopwords.txt")
        self.corpus_file = corpus_file
        self.outfile = outfile
        self.get_train_format_file()

    def get_train_format_file(self):
        print(">>>>> preprocess corpus")
        data = self.get_train_format_data(category_level=1)
        self.write_txt_file(data, self.outfile)

    def get_train_format_data(self, category_level=1):
        data = list()
        for line in read_json_format_file(self.corpus_file):
            if line:
                (_id, channel, title, content), y = self._preline(line)
                label = y[category_level - 1]
                _, title_char_feature, title_token_feature = self.clean_doc_obj.get_char_and_token_feature(title)
                _, content_char_feature, content_token_feature = self.clean_doc_obj.get_char_and_token_feature(content)

                # 获取title和content的char特征
                char_feature = " ".join(list(title_char_feature)) + " ### " + " ".join(
                    list(content_char_feature))
                # 获取title和content的token特征
                token_feature = " ".join(list(title_token_feature)) + " ### " + " ".join(
                    list(content_token_feature))
                data.append(label + "\t" + token_feature + "\t" + char_feature + "\t" + channel)
            else:
                continue
        return data

    @staticmethod
    def _preline(line):
        """
        处理文件行（dict格式）
        """
        article_id = line.get("id", "")
        channel = ""
        title = line.get("title", "")
        content = line.get("content", "")
        top_category = line.get("manual_tag-3.0", "")
        sub_category = line.get("manual_subtag-3.0", "")
        third_category = ""

        return (article_id, channel, title, content), (top_category, sub_category, third_category)

    def write_txt_file(self, data, file):
        """
        写数据到文件
        """
        print(">>>>> start writing file")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< write down：{}".format(file))


class PrePublicCorpus(object):
    """
    将公开数据集处理为模型训练数据
    """

    def __init__(self, corpus_file, out_dir):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        self.clean_doc_obj = CleanDoc()
        self.corpus_file = corpus_file
        self.train_file = os.path.join(out_dir, "train.txt")
        self.dev_file = os.path.join(out_dir, "validation.txt")
        self.test_file = os.path.join(out_dir, "test.txt")
        # self.analysis_label_dist()
        self.get_category_corpus_file()

    def get_category_corpus_file(self):
        print(">>>>> preprocess corpus")
        X, Y = self.get_clean_data(category_level=1)
        train, dev, test = self.stratified_sampling(X, Y, 0.2)
        self.write_txt_file(train, self.train_file)
        self.write_txt_file(dev, self.dev_file)
        self.write_txt_file(test, self.test_file)

    def get_clean_data(self, category_level=1):
        X = list()
        Y = list()
        _count = dict()
        for line in self.read_json_format_file(self.corpus_file):
            if line:
                (_id, channel, title, content), y = self._preline(line)
                label = y[category_level - 1]

                _, title_char_feature, title_token_feature = self.clean_doc_obj.get_char_and_token_feature(title)
                _, content_char_feature, content_token_feature = self.clean_doc_obj.get_char_and_token_feature(content)

                # 获取title和content的char特征
                char_feature = " ".join(list(title_char_feature)) + " ### " + " ".join(
                    list(content_char_feature))
                # 获取title和content的token特征
                token_feature = " ".join(list(title_token_feature)) + " ### " + " ".join(
                    list(content_token_feature))

                if len(title) > 10:
                    if label in _count.keys():
                        if _count[label] > 10000:
                            continue
                        else:
                            _count[label] += 1
                            X.append(token_feature + "\t" + char_feature + "\t" + channel)
                            Y.append(label)
                    else:
                        _count[label] = 1
                        X.append(token_feature + "\t" + char_feature + "\t" + channel)
                        Y.append(label)
                else:
                    continue
        return X, Y

    def stratified_sampling(self, x, y, valid_portion, validation=True):
        """
        按标签类别个数分层切分训练集,验证集,测试集
        :param x:
        :param y:
        :param valid_portion:
        :return:
        """
        skf = StratifiedKFold(n_splits=int(1 / valid_portion))
        train = None
        dev = None
        test = None

        index = [(train_index, dev_test_index) for train_index, dev_test_index in skf.split(x, y)]

        train_label_count = self._label_count([y[i] for i in index[0][0]])
        print("train_label_count: {}".format(json.dumps(train_label_count, indent=4, ensure_ascii=False)))
        train = [y[i] + "\t" + x[i] for i in index[0][0]]
        if validation:
            dev_test_skf = StratifiedKFold(n_splits=2)
            dev_test_x = [x[j] for j in index[0][1]]
            dev_test_y = [y[j] for j in index[0][1]]
            dev_test_index = [(dev_index, test_index) for dev_index, test_index in dev_test_skf.split(dev_test_x, dev_test_y)]
            validation_label_count = self._label_count([y[i] for i in dev_test_index[0][0]])
            print("validation_label_count: {}".format(json.dumps(validation_label_count, indent=4, ensure_ascii=False)))
            test_label_count = self._label_count([y[j] for j in dev_test_index[0][1]])
            print("test_label_count: {}".format(json.dumps(test_label_count, indent=4, ensure_ascii=False)))
            dev = [dev_test_y[j] + "\t" + dev_test_x[j] for j in dev_test_index[0][0]]
            test = [dev_test_y[j] + "\t" + dev_test_x[j] for j in dev_test_index[0][0]]
        else:
            dev = [y[j] + "\t" + x[j] for j in index[0][1]]
            test = dev

        return train, dev, test

    def _label_dist_count(self, level1=None, level2=None, level3=None, dist_count=None):
        """
        统计标签分布计算
        :param level1:一级标签
        :param level2:二级标签
        :param level3:三级标签
        :param dist_count:标签分布字典
        :return:
        """
        if level1:
            if level1 in dist_count:
                dist_count[level1]["count"] += 1
            else:
                dist_count[level1] = dict()
                dist_count[level1]["count"] = 1
            if level2:
                if level2 in dist_count[level1]:
                    dist_count[level1][level2]["count"] += 1
                else:
                    dist_count[level1][level2] = dict()
                    dist_count[level1][level2]["count"] = 1
                if level3:
                    if level3 in dist_count[level1][level2]:
                        dist_count[level1][level2][level3] += 1
                    else:
                        dist_count[level1][level2][level3] = 1

    @staticmethod
    def dict_sort(result, limit_num=None):
        """
        字典排序, 返回有序字典
        :param result:
        :param limit_num:
        :return:
        """
        _result_sort = sorted(result.items(), key=lambda x: x[1]["count"], reverse=True)
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

    @staticmethod
    def _preline(line):
        """
        处理文件行（dict格式）
        """
        article_id = ""
        channel = "thuc_news"
        title = line.get("text", "")
        content = ""
        top_category = line.get("label", "")
        sub_category = ""
        third_category = ""
        return (article_id, channel, title, content), (top_category, sub_category, third_category)

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if label_count.get(i, None) is not None:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    @staticmethod
    def read_json_format_file(file):
        """
        读取每行为json格式的文本
        :param file: 文件名
        :return: 每行文本
        """
        if not os.path.exists(file):
            raise FileNotFoundError("file {} not found.".format(file))
        print(">>>>> reading file：{}".format(file))
        line_count = 0
        with open(file, 'r') as f:
            while True:
                _line = f.readline()
                line_count += 1
                if not _line:
                    break
                else:
                    line = json.loads(_line.strip())
                    # line = eval(_line.strip())
                    if line_count % 100000 == 0:
                        print(">>>>> read {} lines.".format(line_count))
                    yield line

    def write_txt_file(self, data, file):
        """
        写数据到文件
        """
        print(">>>>> start writing file")
        with open(file, "w") as f:
            for line in data:
                f.write(line + "\n")
        print("<<<<< write down：{}".format(file))


def count_label(file):
    label_count = dict()
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            label = data[0]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

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

    print("label_count: {}".format(json.dumps(dict_sort(label_count), ensure_ascii=False, indent=4)))


def pre_public_process():
    data_file = "./test_data/thuc_news_sample"
    out_dir = "./test_data"
    PrePublicCorpus(data_file, out_dir)


def pre_process():
    data_file = "/work/data/corpus_data"
    out_dir = "./data"
    PreCorpus(data_file, out_dir)

def pre_large_label_process():
    train_file = "data/train.txt"
    train_large_label_file = "data/large_label_train.txt"
    val_file = "data/validation.txt"
    val_large_label_file = "data/large_label_validation.txt"
    test_file = "data/test.txt"
    test_large_label_file = "data/large_label_test.txt"
    PreLargeLabelCorpus(train_file, train_large_label_file)
    PreLargeLabelCorpus(val_file, val_large_label_file)
    PreLargeLabelCorpus(test_file, test_large_label_file)


def get_label_dist():
    train_file = "./data/train.txt"
    test_file = "./data/test.txt"
    count_label(train_file)
    count_label(test_file)


def pre_process_v1():
    file = "/work/data/category/corpus/20200615180704_trec_train.txt"
    outfile = "data/train.txt"
    PreCorpusV1(file, outfile)
    val_file = "/work/data/category/corpus/20200615180723_trec_test.txt"
    val_outfile = "data/validation.txt"
    PreCorpusV1(val_file, val_outfile)


def pre_test_process():
    data_file = "data/tag3.0_test.txt"
    out_file = "data/test.txt"
    PreTestCorpus(data_file, out_file)


def pre_test_single_label_process():
    data_file = "data/test.txt"
    out_file = "data/test_single_label.txt"
    with open(data_file, "r", encoding="utf-8") as f, \
        open(out_file, "w", encoding="utf-8") as wf:
        for line in f.readlines():
            label = line.strip().split("\t")[0].split(",")
            if len(label) == 1:
                wf.write(line)



def clean_text_demo():
    title = "搞笑GIF：聪明人一看就知道怎么做到的！"
    content = "媳妇我不练了，不练了！ 哎呀，悲催了 大哥，看着点路 一个月了原来我的篮球在这里？ 不懂艺术的人们 怪我喽？ 白猫：你干嘛，干嘛，干嘛…… 真是人间美味啊！ 妹子，还以为你衣服穿反了 你会手指打结吗？ 美女，你这喉结过分了啊 聪明人一看就知道怎么做到的！ 面这两人也是躺着中枪 辣条都让你们玩涨价了！ 妹子砸车的威武霸气，真是惹不起 这手势太吓人了，我要下车！ 妹子都进电梯了，你还拉她出来干啥 妹子这么勤快呢，桌子很干净了，休息会吧 看得出，教练确实很受打击 去车展的有几个是卖车的，我感觉更多的人是看美女的啊！ 讲道理，现在什么都是智能，连沙袋都变成智能的了 都捏不碎，老丈人表示不服，非要亲自示范一下 跟老师们一起吃饭，感觉很开心！ 爱她就背起她走回家，妥妥的真爱呐 对女孩子来说，头发就是的她的命 长不大的妹子，连鬼你都戏弄 好玩 大哥你配合得太专业了啊 知道篮球场汤为什么总是会有这么多大神呢？你这里比较有市场需求 兄弟，现在的日子过的是越来越好了啊 啊！我的眼睛"
    clean_doc_obj = CleanDoc(stopwords_file="./data/stopwords.txt")
    c_title, title_char, title_token = clean_doc_obj.get_char_and_token_feature(title)
    print(c_title)
    print(title_char)
    c_content, content_char, content_token = clean_doc_obj.get_char_and_token_feature(content)
    print(c_content)
    print(content_token)


def main():
    pre_public_process()
    # pre_process()
    # pre_process_v1()
    # pre_large_label_process()
    # clean_text_demo()
    # pre_test_process()
    # pre_test_single_label_process()


if __name__ == "__main__":
    main()