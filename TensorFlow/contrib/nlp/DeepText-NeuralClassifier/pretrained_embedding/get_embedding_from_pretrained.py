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

import codecs
from tqdm import tqdm
from gensim.models import KeyedVectors

def get_pretrained_embedding(pretrained_embedding_file):
    # wv_from_text = KeyedVectors.load_word2vec_format(pretrained_embedding_file, binary=False)  # 加载时间比较长
    pretrain_embedding = dict()
    with open(pretrained_embedding_file, 'r', encoding='utf-8', errors='ignore')as f:
        for i in tqdm(range(8824330)):  # 似乎不同时期下载的词向量range并不一样
            data = f.readline()
            a = data.split()
            if i == 0:
                continue
            if len(a) == 201:
                if not a[0].isdigit():
                    n = n + 1
                    pretrain_embedding[a[0]] = a[1:]
    return pretrain_embedding


def write_embedding_to_file(feature_file, out_embedding_file, pretrained_embedding):
    with open(feature_file, "r", encoding="utf-8") as f, \
            open(out_embedding_file, "w", encoding="utf-8") as wf:
        lines = f.readlines()
        for line in lines:
            word = line.strip().split("\t")[0]
            if word in pretrained_embedding:
                wf.write("{}\t{}\n".format(word, " ".join(pretrained_embedding[word])))


def main():
    tencent_pretrain_file = "/data/word_embedding/Tencent_AILab_ChineseEmbedding.txt"
    char_dict = "classifier_result/dict/dict_main/char.dict"
    token_dict = "classifier_result/dict/dict_main/token.dict"
    token_pretrain_file = "classifier_result/pretrained_embedding/token_pretrained_embedding.txt"
    char_pretrain_file = "classifier_result/pretrained_embedding/char_pretrained_embedding.txt"
    pretrained_embedding = get_pretrained_embedding(tencent_pretrain_file)
    write_embedding_to_file(char_dict, char_pretrain_file, pretrained_embedding)
    write_embedding_to_file(token_dict, token_pretrain_file, pretrained_embedding)


if __name__ == "__main__":
    main()
