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

import re
import os
import string
import numpy as np
import jieba
import emoji



class CleanDoc(object):
    """
    文本清洗并进行特征抽取
    """
    def __init__(self, language="cn", stopwords_file="stopwords.txt"):

        if os.path.exists(stopwords_file):
            with open(stopwords_file, "r", encoding="utf-8") as fr:
                self.stopwords = set([line.strip() for line in fr.readlines()])
        else:
            self.stopwords = set()
        self.language = language

    def get_char_and_token_feature(self, sentence):
        """
        获取文本的char和token特征
        :param sentence: 原始文本
        :return: 清洗后的文本，char特征，token特征
        """
        if self.language == "cn":
            # text = self.clean_cn_text(sentence)
            clean_text = self.clean_cn_text(sentence)
            char_feature = self.get_cn_char_feature(clean_text)
            token_feature = self.get_cn_token_feature(clean_text, remove_stopwords=True)
            # char_feature = self.get_cn_char_feature_v1(clean_text)
            # token_feature = self.get_cn_token_feature_v1(clean_text)
        else:
            clean_text = self.clean_en_text(sentence)
            char_feature = self.get_en_char_feature(clean_text)
            token_feature = self.get_en_token_feature(clean_text)
        return clean_text, char_feature, token_feature

    def clean_cn_text(self, sentence):
        """
        中文文本清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 清洗网址
        step3 -> 清洗邮箱
        step4 -> 清洗表情等非英文字符
        step5 -> 替换多个空格为一个空格
        :param sentence: 原始文本
        :return: 清洗后的文本
        """

        _text = sentence.replace('\u2028', '').replace('\n', '').replace('\t', '')
        re_h = re.compile('<(/?\w+|!--|!DOCTYPE|\?xml)[^>]*>')
        _text = re_h.sub('', _text)  # html处理
        no_html = self.clean_url(_text)
        no_mail = self.clean_mail(no_html)
        no_emoji = self.remove_emoji(no_mail)
        text = re.sub(r"\s+", " ", no_emoji)
        return text

    def get_cn_char_feature(self, text):
        """
        按字切分句子,去除非中文字符及标点，获取char级别特征
        todo:针对中文字级别处理（针对英文、数字等符号特殊处理,停用词）
        :param text:
        :return:
        """
        # print("splitting chinese char")
        seg_list = list()
        none_chinese = ""
        for char in text:
            if self.is_chinese(char) is False:
                if char in self.punc_list:
                    continue
                none_chinese += char
            else:
                if none_chinese:
                    seg_list.append(none_chinese)
                    none_chinese = ""
                seg_list.append(char)
        if not seg_list:
            seg_list = list()
        return seg_list

    def get_cn_token_feature(self, text, remove_stopwords=False):
        """
        按结巴分词（默认精确模式）
        去除标点，去除停用词（可选），获取文本token级别特征
        :param text: 清洗干净的文本
        :param remove_stopwords: 是否去除停用词
        :return:
        """
        # 精确模式
        seg_list = jieba.cut(text, cut_all=False)
        words = [w for w in seg_list if w not in self.punc_list]

        # 如果传入了停用词表，则去除停用词
        if remove_stopwords and self.stopwords :
            words = [word for word in words if word not in self.stopwords]

        if not words:
            words = list()

        return words

    def clean_cn_text_v1(self, sentence):
        """
        版本v1提供的方法@刘欢
        用第三方库清洗中文文本
        """
        from harvesttext import HarvestText
        ht_obj = HarvestText()
        # 去掉微博的@，表情符；网址；email；html代码中的一类的特殊字符等
        _text = sentence.replace('\u2028', '').replace('\n', '').replace('\t', '')
        re_h = re.compile("<(/?\w+|!--|!DOCTYPE|\?xml)[^>]*>")
        _text = re_h.sub("", _text)  # html处理
        clean_text = ht_obj.clean_text(_text)
        return clean_text

    def get_cn_token_feature_v1(self, text):
        """
        版本v1提供的方法@刘欢
        """
        if text is np.nan:
            text = ''
        return list(jieba.cut(text))

    def get_cn_char_feature_v1(self, text):
        """
        版本v1提供的方法@刘欢
        """
        if text is np.nan:
            text = ''
        return list(text)

    def clean_en_text(self, sentence):
        """
        英文文本清洗流程
        step1 -> 替换掉换行符、制表符等
        step2 -> 转小写
        step3 -> 清洗网址
        step4 -> 清洗邮箱
        step5 -> 清洗表情等非英文字符
        step6 -> 清洗标点符号、数字
        step7 -> 替换多个空格为一个空格
        :param sentence: 原始文本
        :return: 清洗后的文本
        """
        text = sentence.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        _text = text.lower()
        no_html = self.clean_url(_text)
        no_mail = self.clean_mail(no_html)
        no_emoji = self.remove_emoji(no_mail)
        no_symbol = self.remove_symbol(no_emoji)
        text = re.sub(r"\s+", " ", no_symbol)
        return text

    def get_en_char_feature(self, sentence):
        """
        获取英文字符级别特征
        :param sentence: 清洗后的文本
        :return: 英文字符级别特征，list
        """
        return " ".join(list(sentence))

    def get_en_token_feature(self, sentence):
        """
        获取英文单词级别特征
        :param sentence: 清洗后的文本
        :return: 英文单词级别特征，list
        """
        return sentence

    @property
    def punc_list(self):
        """
        标点符号，包括半角，全角符号
        """
        add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：”“^-——=&#@￥\n「」…『』\u3000\xa0'
        return string.punctuation + add_punc

    @staticmethod
    def remove_en_emoji(text):
        """
        去除英文表情符号
        :param text:
        :return:
        """
        cleaned_text = ""
        for c in text:
            if (ord(c) >= 65 and ord(c) <= 126) or (ord(c) >= 32 and ord(c) <= 63):
                cleaned_text += c
        return cleaned_text

    @staticmethod
    def remove_emoji(text):
        """
        去除表情符号
        :param text:
        :return:
        """
        token_list = text.replace("¡", "").replace("¿", "").split(" ")
        em_str = r":.*?:"
        em_p = re.compile(em_str, flags=0)
        clean_token = list()
        for token in token_list:
            em = emoji.demojize(token)
            emj = em_p.search(em)
            if emj:
                _e = emj.group(0)
                # print(_e)
            else:
                clean_token.append(token)
        cleaned_text = " ".join(clean_token)
        return cleaned_text.strip()

    @staticmethod
    def is_chinese(uchar):
        """
        判断一个unicode是否是汉字
        """
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False

    @staticmethod
    def clean_url(text):
        """
        去除网址
        1.完整网址https开头的
        2.没有协议头的网址，www开头的
        :param text:
        :return:
        """

        pattern = re.compile(
            r'(?:(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])|(?:www\.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])')
        # pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-zA-Z][0-9a-zA-Z]))+')
        # url_list = re.findall(pattern, text)
        # for url in url_list:
        #     text = text.replace(url, " ")
        text = pattern.sub("", text)
        return text.replace("( )", " ")

    @staticmethod
    def clean_mail(text):
        """
        去除邮箱
        :param text:
        :return:
        """
        pattern = re.compile(r"\w+[-_.]*[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}")
        text = pattern.sub(" ", text)
        # mail_list = re.findall(pattern, text)
        # for mail in mail_list:
        #     text = text.replace(mail, " ")
        return text

    @staticmethod
    def remove_symbol_and_digits(text):
        """
        去除标点符号和数字
        :param text:
        :return:
        """
        del_symbol = string.punctuation + string.digits  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text

    @staticmethod
    def remove_symbol(text):
        """
        去除标点符号
        :param text:
        :return:
        """
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = text.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        return text