#
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
#
from npu_bridge.npu_init import *
from char_sim import CharFuncs
from masked_lm import MaskedLM
from bert_modified import modeling
import re
import json
import pickle
import argparse
import numpy
import logging
import plot
import tqdm
import time
# import npu_device
# print('npu_device loaded')
# npu_device.global_options().precision_mode = 'allow_fp32_to_fp16'
# npu_config = {}
# npu_device.open().as_default()




####################################################################################################

__author__ = 'Yuzhong Hong <hongyuzhong@qiyi.com / eugene.h.git@gmail.com>'
__date__ = '10/09/2019'
__description__ = 'The main script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker'

#鍔犺浇faspell鐨勯厤缃枃浠
CONFIGS = json.loads(open('faspell_configs.json', 'r', encoding='utf-8').read())
#浠庨厤缃俊鎭腑鑾峰彇 褰技 鍜岄煶浼肩殑鏉冮噸
WEIGHTS = (CONFIGS["general_configs"]["weights"]["visual"], CONFIGS["general_configs"]["weights"]["phonological"], 0.0)

#浠庨厤缃俊鎭腑鑾峰彇閫氱敤閰嶇疆淇℃伅
CHAR = CharFuncs(CONFIGS["general_configs"]["char_meta"])


#浠庨厤缃俊鎭腑鑾峰彇MLM鐨勭浉鍏抽厤缃俊鎭
class LM_Config(object):
    max_seq_length = CONFIGS["general_configs"]["lm"]["max_seq"]
    vocab_file = CONFIGS["general_configs"]["lm"]["vocab"]
    bert_config_file = CONFIGS["general_configs"]["lm"]["bert_configs"]
    #閫夋嫨寰皟鍚庣殑妯″瀷杩樻槸閫夋嫨鍘熺敓BERT
    if CONFIGS["general_configs"]["lm"]["fine_tuning_is_on"]:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["fine-tuned"]
    else:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["pre-trained"]
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    topn = CONFIGS["general_configs"]["lm"]["top_n"]


#鏋勯犺繃婊ゅ櫒
class Filter(object):
    def __init__(self):
        self.curve_idx_sound = {0: {0: Curves.curve_null,#Curves.curve_full,  # 0 for non-difference
                              1: Curves.curve_null,#Curves.curve_d0r1p,
                              2: Curves.curve_null,#Curves.curve_d0r2p,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              },
                          1: {0: Curves.curve_null,#Curves.curve_d1r0p,  # 1 for difference
                              1: Curves.curve_null,#Curves.curve_d1r1p,
                              2: Curves.curve_null,#Curves.curve_d1r2p,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              }}

        self.curve_idx_shape = {0: {0: Curves.curve_null,#Curves.curve_full,  # 0 for non-difference
                                    1: Curves.curve_null,#Curves.curve_d0r1s,
                                    2: Curves.curve_null,#Curves.curve_d0r2s,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    },
                                1: {0: Curves.curve_null,#Curves.curve_d1r0s,  # 1 for difference y1 = (7.64960918 * x1 -7) / - 2.87156076
                                    1: Curves.curve_null,#Curves.curve_d1r1s,
                                    2: Curves.curve_null,#Curves.curve_d1r2s,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    }}

    def filter(self, rank, difference, error, filter_is_on=True, sim_type='shape'):
        if filter_is_on:#寮鍚繃婊ゆ潯浠朵笅 鏍规嵁鍊欓夊瓧杩涜杩囨护绫诲埆閫夋嫨 鏄惁灞炰簬top-difference -> 灞炰簬鍝竴涓猺ank -> 灞炰簬sound杩樻槸shape
            if sim_type == 'sound':
                curve = self.curve_idx_sound[int(difference)][rank]
            else:
                # print(int(difference))
                curve = self.curve_idx_shape[int(difference)][rank]
        else:
            curve = Curves.curve_null

        if curve(error["confidence"], error["similarity"]) and self.special_filters(error):
            return True

        return False

    #鍔犲叆瑙勫垯杩涜杩囨护
    @staticmethod
    def special_filters(error):
        """
        Special filters for, essentially, grammatical errors. The following is some examples.
        """
        # if error["original"] in {'浠': 0, '濂': 0, '浣': 0, '濡': 0}:
        #     if error["confidence"] < 0.95:
        #         return False
        #
        # if error["original"] in {'鐨': 0, '寰': 0, '鍦': 0}:
        #     if error["confidence"] < 0.6:
        #         return False
        #
        # if error["original"] in {'鍦': 0, '鍐': 0}:
        #     if error["confidence"] < 0.6:
        #         return False

        return True


#杩囨护鍣ㄤ腑搴旂敤鐨勮繃婊ゆ洸绾
class Curves(object):
    def __init__(self):
        pass

    #涓嶈繘琛岃繃婊
    @staticmethod
    def curve_null(confidence, similarity):
        """This curve is used when no filter is applied"""
        return True
    #杩囨护鎵鏈
    @staticmethod
    def curve_full(confidence, similarity):
        """This curve is used to filter out everything"""
        return False

    #杩欏氨鏄鏂囦腑鎵閫夋嫨鐨勪汉宸ョ‘瀹氱殑鏇茬嚎杩囨护鏂瑰紡锛屽彲浠ラ噰鐢ㄥ鏍圭洿绾胯繘琛岃仈鍚堣繃婊 妯℃嫙鏇茬嚎鏁堟灉
    @staticmethod
    def curve_01(confidence, similarity):
        """
        we provide an example of how to write a curve. Empirically, curves are all convex upwards.
        Thus we can approximate the filtering effect of a curve using its tangent lines.
        """
        #浣跨敤涓ゆ牴鐩寸嚎杩涜杩囨护
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0
        if flag1 or flag2:
            return True

        return False
    #========================================================================#
    def curve_d1r0s(confidence, similarity):
        flag1 = 1 * confidence + similarity * 1 -1> 0
        flag2 = similarity > 0.4
        if flag1 or flag2:
            return True
        return False

    def curve_d1r1s(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.5 -0.3 > 0
        if flag1:
            return True

        return False

    def curve_d1r2s(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.4 -0.24 > 0
        if flag1:
            return True

        return False

    def curve_d1r0p(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.8 -0.48 > 0
        flag2 = similarity > 0.3
        if flag1 and flag2:
            return True

        return False

    def curve_d1r1p(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.6 -0.36 > 0
        flag2 = similarity > 0.3
        if flag1 and flag2:
            return True

        return False

    def curve_d1r2p(confidence, similarity):
        flag1 = 0.8 * confidence + similarity*0.1 -0.08 > 0
        flag2 = similarity > 0.6
        if flag1 and flag2:
            return True

        return False

        return False

    def curve_d0r1s(confidence, similarity):
        flag1 = 0.8 * confidence + similarity*0.4 -0.32 > 0
        flag2 = similarity > 0.4
        flag3 = similarity < 0.8
        if flag1 and flag2 and flag3:
            return True

        return False

    def curve_d0r1p(confidence, similarity):
        flag1 = 0.9* confidence + similarity*0.7 -0.63 > 0
        flag2 = similarity > 0.7

        if flag1 and flag2:
            return True

        return False

    def curve_d0r2p(confidence, similarity):
        flag1 = 1* confidence + similarity*0.4 -0.4 > 0
        flag2 = similarity > 0.7

        if flag1 and flag2:
            return True

        return False

    def curve_d0r2s(confidence, similarity):
        flag1 = 1 * confidence + similarity*0.4 -0.4 > 0
        flag2 = similarity > 0.4

        if flag1 and flag2:
            return True

        return False
    # ========================================================================#
#鏋勯燙SCChecker 鐢变袱閮ㄥ垎缁勬垚 MLM 鍜 杩囨护鍣
class SpellChecker(object):
    def __init__(self):
        self.masked_lm = MaskedLM(LM_Config()) #MLM
        self.filter = Filter() #杩囨护鍣

    #娣诲姞閮ㄥ垎瑙勫垯 纭畾鍝簺绾犳琚涓烘棤鏁
    @staticmethod
    def pass_ad_hoc_filter(corrected_to, original): #original鍘熷瀛楃锛堣緭鍏ワ級 corrected_to绾犳鍚庣殑瀛楃
        if corrected_to == '[UNK]':
            return False
        if '#' in corrected_to:
            return False
        if len(corrected_to) != len(original):
            return False
        if re.findall(r'[a-zA-Z锛-锛猴絹-锝歖+]', corrected_to):
            return False
        if re.findall(r'[a-zA-Z锛-锛猴絹-锝歖+]', original):
            return False
        return True

    #閿欒瀛楃鐨勬鏌
    def get_error(self, sentence, j, cand_tokens, rank=0, difference=True, filter_is_on=True, weights=WEIGHTS, sim_type='shape'):
        """
        PARAMS
        ------------------------------------------------
        sentence: sentence to be checked
        j: position of the character to be checked
        cand_tokens: all candidates
        rank: the rank of the candidate in question
        filters_on: only used in ablation experiment to remove CSD
        weights: weights for different types of similarity
        sim_type: type of similarity

        """

        cand_token, cand_token_prob = cand_tokens[rank]

        if cand_token != sentence[j]:#鍊欓夊拰鍘熷涓嶄竴鏍 鍒ゆ柇鏄惁涓烘湁鏁堢殑閿欒妫娴
            #瀵逛簬char鐨別rror鍏蜂綋缁撴瀯
            error = {"error_position": j,
                     "original": sentence[j],
                     "corrected_to": cand_token,
                     "candidates": dict(cand_tokens),
                     "confidence": cand_token_prob,
                     "similarity": CHAR.similarity(sentence[j], cand_token, weights=weights),#weights琛ㄧず鍦ㄨ绠楃浉浼煎害鏃 涓嶅悓閮ㄥ垎鎵鍗犳瘮渚 锛坰hape, sound, freq锛
                     "sentence_len": len(sentence)}

            if not self.pass_ad_hoc_filter(error["corrected_to"], error["original"]):
                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None #鏃犳晥鐨勯敊璇

            else:
                if self.filter.filter(rank, difference, error, filter_is_on, sim_type=sim_type):
                    logging.info(f'{error["original"]}'
                                 f'--> {error["corrected_to"]}'
                                 f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                    return error #鏈夋晥鐨勯敊璇

                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None #鏃犳晥鐨勯敊璇

        logging.info(f'{sentence[j]}'
                     f' --> <PASS-{sentence[j]}>'
                     f' (con={cand_token_prob}, sim=null, on_top_difference={difference})')
        return None #娌℃湁閿欒 鍊欓夌殑鍜屽師濮嬬殑瀛楃涓鏍

    #閿欒鐨勭籂姝
    def make_corrections(self,
                         sentences,
                         tackle_n_gram_bias=CONFIGS["exp_configs"]["tackle_n_gram_bias"],
                         rank_in_question=CONFIGS["general_configs"]["rank"],
                         dump_candidates=CONFIGS["exp_configs"]["dump_candidates"],
                         read_from_dump=CONFIGS["exp_configs"]["read_from_dump"],
                         is_train=False,
                         train_on_difference=True,
                         filter_is_on=CONFIGS["exp_configs"]["filter_is_on"],
                         sim_union=CONFIGS["exp_configs"]["union_of_sims"]
                         ):
        """
        PARAMS:
        ------------------------------
        sentences: sentences with potential errors
        tackle_n_gram_bias: whether the hack to tackle n gram bias is used
        rank_in_question: rank of the group of candidates in question
        dump_candidates: whether save candidates to a specific path
        read_from_dump: read candidates from dump
        is_train: if the script is in the training mode
        train_on_difference: choose the between two sub groups
        filter_is_on: used in ablation experiments to decide whether to remove CSD
        sim_union: whether to take the union of the filtering results given by using two types of similarities

        RETURN:
        ------------------------------
        correction results of all sentences
        """
        #瀵瑰緟绾犻敊鍙ュ瓙鍋氶澶勭悊 鍓嶅悗娣诲姞銆
        processed_sentences = self.process_sentences(sentences)
        generation_time = 0
        if read_from_dump:#浠巇ump鍔犺浇candidate
            assert dump_candidates
            topn_candidates = pickle.load(open(dump_candidates, 'rb'))
        else:#浣跨敤MLM杩涜candidate鐨勭敓鎴
            start_generation = time.time()
            topn_candidates = self.masked_lm.find_topn_candidates(processed_sentences,
                                                                  batch_size=CONFIGS["general_configs"]["lm"][
                                                                      "batch_size"])
            end_generation = time.time()
            generation_time += end_generation - start_generation #鐢熸垚condidate鎵鐢ㄧ殑鏃堕棿
            if dump_candidates:#鍐欏叆dump鏂囦欢 闃叉鍦ㄨ缁僀SD鐨勬椂鍊欓噸澶嶈皟鐢ㄦā鍨嬭繘琛岀敓鎴
                pickle.dump(topn_candidates, open(dump_candidates, 'wb'))

        # main workflow  CSC宸ヤ綔鐨勪富瑕佹祦绋
        filter_time = 0 #filter鎵鐢ㄦ椂闂
        skipped_count = 0 #skip娆℃暟
        results = []    #璁板綍瀵箂entences鐨凜SC缁撴灉
        print('making corrections ...')
        if logging.getLogger().getEffectiveLevel() != logging.INFO:  # show progress bar if not in verbose mode
            progess_bar = tqdm.tqdm(enumerate(topn_candidates))
        else:
            progess_bar = enumerate(topn_candidates)

        for i, cand in progess_bar:#寰幆姣忎竴涓彞瀛愮殑topn_candidate
            logging.info("*" * 50)
            logging.info(f"spell checking sentence {sentences[i]}")
            sentence = '' #瀛樺偍CSC鍚庣殑鍙ュ瓙
            res = [] #瀛樺偍閿欒淇℃伅 姣忎竴涓敊璇俊鎭殑缁撴瀯鏄竴涓猚har鐨別rror

            # can't cope with sentences containing Latin letters yet.
            # 杩囨护閭ｄ簺鍚湁鎷変竵鏂囧瓧姣嶇殑鍙ュ瓙
            if re.findall(r'[a-zA-Z锛-锛猴絹-锝歖+]', sentences[i]):#瀵瑰緟check 鍙ュ瓙 杩涜閫愪竴check
                skipped_count += 1
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentences[i],
                                "num_errors": 0,
                                "errors": []
                                })
                logging.info("containing Latin letters; pass current sentence.")

            else:

                # when testing on SIGHAN13,14,15, we recommend using `extension()` to solve
                # issues caused by full-width humbers;
                # when testing on OCR data, we recommend using `extended_cand = cand`
                #extended_cand = extension(cand)
                extended_cand = cand
                for j, cand_tokens in enumerate(extended_cand):  # spell check for each characters 瀵逛簬姣忎竴涓緟妫鏌ョ殑鍙ュ瓙閫愪竴妫鏌ユ瘡涓猚har
                    if 0 < j < len(extended_cand) - 1:  # skip the head and the end placeholders -- `銆俙 鍥犱负鍦ㄥ彞瀛愮殑棣栧熬鍔犱簡銆
                        # print(j)

                        char = sentences[i][j - 1]

                        # detect and correct errors
                        error = None

                        # spell check rank by rank 鎸夌収rank鐨勬帓琛岄愪竴妫鏌  rank_in_question瀹炲湪faspellconfig.json涓寚瀹氱殑
                        start_filter = time.time()
                        for rank in range(rank_in_question + 1):
                            logging.info(f"spell checking on rank={rank}")

                            if not sim_union:#瀵圭被鍨嬬殑涓嶅悓閫夋嫨杩涜鍒嗗紑妫鏌 璁＄畻sim鐨勬椂鍊檞eight涓篬1,0,0] 鎴栬匸0,1,0]
                                if WEIGHTS[0] > WEIGHTS[1]:#褰㈢姸鐩镐技
                                    sim_type = 'shape'
                                else:#鍙戦煶鐩镐技
                                    sim_type = 'sound'
                                error = self.get_error(sentences[i],#瀵圭i涓彞瀛愯繘琛屾鏌
                                                       j - 1,#妫瀵熺i鍙ュ瓙涓殑绗琷-1涓瓧绗
                                                       cand_tokens,#绗琲涓彞瀛愮j-1涓猚har鐨刢andidate
                                                       rank=rank,#閫愪竴閫夋嫨rank
                                                       difference=cand_tokens[0][0] != sentences[i][j - 1],#鍒ゆ柇鎺掑悕绗竴鐨勫欓夊瓧绗︿笌鍘熷瀛楃鏄惁涓嶅悓
                                                       filter_is_on=filter_is_on, sim_type=sim_type)

                            else:#瀵逛袱绉嶇被鍨嬪潎杩涜妫鏌

                                logging.info("using shape similarity:")
                                error_shape = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(1, 0, 0), sim_type='shape')
                                logging.info("using sound similarity:")
                                error_sound = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(0, 1, 0), sim_type='sound')
  

                                if error_shape:
                                    error = error_shape
                                    if is_train:
                                        error = None  # to train shape similarity, we do not want any error that has already detected by sound similarity
                                                        #鎴戣涓鸿繖閲岀殑閫昏緫鍐欑殑鏈夌偣闂 鎵浠ラ伩鍏嶈繖涓棶棰樺奖鍝嶇粨鏋 鍦ㄨ缁僀SD鐨勬椂鍊 淇濇寔union_of_sims涓篺alse
                                else:
                                    error = error_sound

                            #褰撳湪璇ヨ疆rank涓壘鍒颁簡閿欒鐨勬椂鍊
                            if error:
                                if is_train:#璁粌CSD妯″紡涓
                                    if rank != rank_in_question:  # not include candidate that has a predecessor already
                                        # taken as error
                                        error = None
                                        # break
                                    else:
                                        # do not include candidates produced by different candidate generation strategy#鍦ㄤ袱涓瓙group涓夋嫨 浣嗘槸涓嶆槑鐧戒负浠涔堣鐢紒=閫昏緫 涓嶇洿鎺ョ敤==
                                        if train_on_difference != (cand_tokens[0][0] != sentences[i][j - 1]):
                                            error = None
                                else:#闈炶缁冩ā寮忎笅
                                    break


                        end_filter = time.time()
                        # 妫鏌ユ瘡涓猚har鍦ㄤ笉鍚宺ank涓墍鐢ㄧ殑鏃堕棿涔嬪拰 涓烘墍鏈塁SC鍙ュ瓙鎵鐢ㄦ椂闂
                        filter_time += end_filter - start_filter

                        #瀵逛簬鎵惧埌浜嗛敊璇殑char灏嗗叾鏇挎崲涓虹籂姝ｅ悗鐨則oken
                        if error:
                            res.append(error)
                            char = error["corrected_to"]
                            sentence += char
                            continue
                        #瀵逛簬涓嶈涓烘槸閿欒鐨刢har 鍒欎粛鐒舵槸鍘熷鐨勫瓧绗
                        sentence += char

                # a small hack: tackle the n-gram bias problem: when n adjacent characters are erroneous,
                # pick only the one with the greatest confidence.
                #瀵逛簬瀹屾暣csc鍚庣殑涓涓彞瀛愮户缁仛浠ヤ笅澶勭悊  杩欏叾瀹炴槸涓涓皬鐨勬妧宸 瀵逛簬杩炵画閿欑殑char 閫夋嫨鍏朵腑鍏锋湁鏈楂榗onfidence鐨
                #閿欒char浣滀负鏄敊璇殑鍒ゅ畾鍜岀籂姝 涔嬫墍浠ヨ杩欐牱鍋氱殑鍘熷洜杩樻湁寰呮帰绌
                error_delete_positions = []
                if tackle_n_gram_bias:
                    error_delete_positions = []
                    for idx, error in enumerate(res):
                        delta = 1
                        n_gram_errors = [error]
                        try:
                            while res[idx + delta]["error_position"] == error["error_position"] + delta:
                                n_gram_errors.append(res[idx + delta])
                                delta += 1
                        except IndexError:
                            pass
                        n_gram_errors.sort(key=lambda e: e["confidence"], reverse=True)
                        error_delete_positions.extend([(e["error_position"], e["original"]) for e in n_gram_errors[1:]])

                    error_delete_positions = dict(error_delete_positions)

                    res = [e for e in res if e["error_position"] not in error_delete_positions]

                    def process(pos, c):
                        if pos not in error_delete_positions:
                            return c
                        else:
                            return error_delete_positions[pos]

                    sentence = ''.join([process(pos, c) for pos, c in enumerate(sentence)])

                # add the result for current sentence
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentence,
                                "num_errors": len(res),
                                "errors": res
                                })
                logging.info(f"current sentence is corrected to {sentence}")
                logging.info(f" {len(error_delete_positions)} errors are deleted to prevent n-gram bias problem")
                logging.info("*" * 50 + '\n')
        try:#CSC鍦╣enerate涓婄殑鏃堕棿鍜宖ilter鐨勬椂闂
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and {filter_time / (len(sentences) - skipped_count) * 1000} ms/sentence in filtering ")
        except ZeroDivisionError:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and NaN ms/sentence in filtering ")
        return results

    #閲嶅杩涜閿欒绾犳 涔熷氨鏄寰匔SC鐨勫彞瀛愯繘琛岄噸澶岰SC
    def repeat_make_corrections(self, sentences, num=3, is_train=False, train_on_difference=True):
        all_results = [] #瀛樺偍鍘嗗彶CSC璁板綍 [[round1],[round2],...,[roundn]]
        sentences_to_be_corrected = sentences

        for _ in range(num):
            results = self.make_corrections(sentences_to_be_corrected,
                                            is_train=is_train,
                                            train_on_difference=train_on_difference)
            sentences_to_be_corrected = [res["corrected_sentence"] for res in results]
            all_results.append(results)
        #閲嶅CSC鐨勫巻鍙茶褰
        correction_history = []
        for i, sentence in enumerate(sentences):
            r = {"original_sentence": sentence, "correction_history": []}
            for item in all_results:#鍦ㄦ瘡涓涓猺ound涓壘绗琲涓彞瀛愮殑csc缁撴灉
                r["correction_history"].append(item[i]["corrected_sentence"])
            correction_history.append(r)

        return all_results, correction_history


    #瀵逛簬杈撳叆鍙ュ瓙鍓嶅悗鍔犱笂銆

    #鍥犱负MLM妯″瀷閮芥槸鍦ㄨ澶氳繛缁彞瀛愪笂杩涜璁粌鐨勶紝鎵浠ュ彞瀛愬紑濮嬪悗缁撴潫鍑烘瀬鏈夊彲鑳借
    #棰勬祴涓衡樸傗欙紙鍋氬疄楠屽彲浠ヨ瀵燂紝纭疄鏄繖鏍凤級銆備负浜嗛伩鍏嶈繖涓棶棰樻病鍦ㄦ瘡涓彞瀛愬墠鍚庡姞涓娾樸傗
    @staticmethod
    def process_sentences(sentences):
        """Because masked language model is trained on concatenated sentences,
         the start and the end of a sentence in question is very likely to be
         corrected to the period symbol (銆) of Chinese. Hence, we add two period
        symbols as placeholders to prevent this from harming FASPell's performance."""

        return ['銆' + sent + '銆' for sent in sentences]


def extension(candidates):#涓や釜閭昏繎鐨刦ull-width 鏁板瓧鎴栧瓧姣嶈褰撳仛涓涓猼oken杈撳叆浜唌lm
    """this function is to resolve the bug that when two adjacent full-width numbers/letters are fed to mlm,
       the output will be merged as one output, thus lead to wrong alignments."""
    new_candidates = []
    for j, cand_tokens in enumerate(candidates): #candidates锛歔[('token1',Pro),('token1',Pro),('token1',Pro)...('token1',Pro)],[('token2',Pro),('token2',Pro),('token2',Pro)...('token2',Pro)]...]
        real_cand_tokens = cand_tokens[0][0]#姣忎竴涓猼oken鐨勭涓涓欓
        if '##' in real_cand_tokens:  # sometimes the result contains '##', so we need to get rid of them first
            real_cand_tokens = real_cand_tokens[2:]
        # 姝ｅ父鎯呭喌 姣忎竴涓骇鐢熺殑鍊欓夐暱搴﹀簲璇ヤ负1  鐢ㄤ簬灏嗛暱搴︿负2鐨勪竴涓猚andidate鎷嗗垎鎴 涓や釜闀垮害涓1鐨刢andidate
        if len(real_cand_tokens) == 2 and not re.findall(r'[a-zA-Z锛-锛猴絹-锝歖+]', real_cand_tokens):
            a = []
            b = []
            for cand, score in cand_tokens:#渚濇澶勭悊姣忎竴涓猼oken鐨勫欓夊瓧
                real_cand = cand
                if '##' in real_cand:
                    real_cand = real_cand[2:]
                a.append((real_cand[0], score))
                b.append((real_cand[-1], score))
            new_candidates.append(a)
            new_candidates.append(b)
            continue

        new_candidates.append(cand_tokens)

    return new_candidates

#鍦ㄦ祴璇曟枃浠朵笂杩涜閲嶅绾犻敊娴嬭瘯
def repeat_test(test_path, spell_checker, repeat_num, is_train, train_on_difference=True):
    sentences = []
    #鏍规嵁鎸囧畾璺緞鑾峰彇鐩稿叧淇℃伅
    for line in open(test_path, 'r', encoding='utf-8'):
        num, wrong, correct = line.strip().split('\t')
        sentences.append(wrong)
    #鑾峰彇鎵鏈夌殑缁撴灉
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=is_train,
                                                                            train_on_difference=train_on_difference)
    if is_train:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path,
                      f'difference_{int(train_on_difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_{i}')
    else:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path, f'test-results_{i}')

    #灏嗙籂姝ｅ巻鍙插啓鍏istory.json鏂囦欢涓
    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()

#绾犻敊娴嬭瘯
def repeat_non_test(sentences, spell_checker, repeat_num):
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=False,
                                                                            train_on_difference=True)

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()

    # 瀵逛簬鍛戒护琛屾墽琛屾ā寮 灏嗙粨鏋滃弽棣堝埌灞忓箷
    args = parse_args()
    if args.mode == 's':
        for i in range(len(correction_history)):
            print('绾犳鍓嶏細', correction_history[i]["original_sentence"])
            print('绾犳鍚庯細', correction_history[i]["correction_history"][-1])
            print('*'*30)

    #灏嗘瘡涓涓彞瀛愮殑绾犻敊缁撴灉鍐欏叆鍒皉esult_{i}.josn涓
    for i, res in enumerate(all_results):
        w = open(f'results_{i}.json', 'w', encoding='utf-8')
        w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
        w.close()



#娴嬭瘯鍗曞厓 鐢ㄩ獙璇丆SC鐨勬晥鏋
def test_unit(res, test_path, out_name, strict=True):
    out = open(f'{out_name}.txt', 'w', encoding='utf-8') #瀛樺叆瀵规瘡涓涓緟CSC鍙ュ瓙鐨勬ц兘妫鏌ョ粨鏋

    corrected_char = 0 #缁熻鍦–SC鍙ュ瓙涓 鎵鏈夌籂姝ｈ繃鐨刢har鏁扮洰
    wrong_char = 0 #缁熻鍦–SC鍙ュ瓙涓 鎵鏈夊瓨鍦ㄩ敊璇殑char鏁扮洰
    corrected_sent = 0
    wrong_sent = 0
    true_corrected_char = 0 #姝ｇ‘绾犳鐨刢har鏁扮洰
    true_corrected_sent = 0
    true_detected_char = 0
    true_detected_sent = 0 #鍙ュ瓙绾у埆 妫閿欐纭殑鍙ュ瓙鏁扮洰
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0


    for idx, line in enumerate(open(test_path, 'r', encoding='utf-8')):
        all_sent += 1
        #浠ヤ笅3涓彉閲 娌℃湁浣跨敤 鍙互娉ㄩ噴鎺
        falsely_corrected_char_in_sentence = 0
        falsely_detected_char_in_sentence = 0
        true_corrected_char_in_sentence = 0

        num, wrong, correct = line.strip().split('\t')
        predict = res[idx]["corrected_sentence"]
        
        wrong_num = 0 #璁板綍涓涓彞瀛愪腑 閿欒绾犳鐨刢har鏁扮洰
        corrected_num = 0 #璁板綍涓涓彞瀛愪腑 绾犳鐨刢har鏁扮洰
        original_wrong_num = 0 #璁板綍涓涓彞瀛愪腑 瀛樺湪閿欒鐨刢har鏁扮洰
        true_detected_char_in_sentence = 0 #璁板綍涓涓彞瀛愪腑姝ｇ‘妫鏌ョ殑char鏁扮洰

        #閫氳繃寰幆鐨勬柟寮 纭畾鍚勪釜鎸囨爣鐨勫叿浣撳
        for c, w, p in zip(correct, wrong, predict):
            if c != p:#姝ｇ‘鐨刢har涓嶇瓑浜庢ā鍨嬭緭鍑虹殑char
                wrong_num += 1 #閿欒绾犳鐨刢har鏁扮洰+1
            if w != p:#鍘熷彞鐨刢har涓嶇瓑浜庢ā鍨嬭緭鍑虹殑char
                corrected_num += 1 #绾犳杩囩殑char娆℃暟+1
                if c == p:#姝ｇ‘绾犳char鏁扮洰+1
                    true_corrected_char += 1
                if w != c:#姝ｇ‘妫鏌har+1 姝ｇ‘妫鏌ュ彞瀛愪腑鐨刢har+1
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if c != w:#姝ｇ‘鐨刢har涓嶇瓑浜庡師濮嬬殑char
                original_wrong_num += 1#鍘熷彞涓璫har閿欒鐨勬暟鐩+1
        #鍐欏叆瀵硅鍙ュ瓙鐨勬ц兘妫鏌ョ粨鏋
        out.write('\t'.join([str(original_wrong_num), wrong, correct, predict, str(wrong_num)]) + '\n')
        corrected_char += corrected_num  #缁熻鎵鏈夌籂姝ｈ繃鐨刢har鏁扮洰
        wrong_char += original_wrong_num #缁熻鎵鏈夊瓨鍦ㄩ敊璇殑char鏁扮洰
        if original_wrong_num != 0:#妫鏌ヨ鍙ュ瓙涓槸鍚︽湁閿欒鐨刢har
            wrong_sent += 1  #閿欒鍙ュ瓙鏁扮洰+1 缁熻鎵鏈夊瓨鍦ㄩ敊璇殑鍙ュ瓙鏁扮洰
        if corrected_num != 0 and wrong_num == 0:#浜岃呭潎婊¤冻鐨勬椂鍊 鍙ュ瓙涓籂姝ｈ繃涓旂籂姝ｅ叏閮ㄦ纭
            true_corrected_sent += 1 #姝ｇ‘绾犳鐨勫彞瀛愭暟鐩+1
        if corrected_num != 0:#鍙弧瓒崇籂姝ｄ簡鍙ュ瓙
            corrected_sent += 1#绾犳鐨勫彞瀛愭暟鐩+1
        if strict:#鏄惁涓ユ牸妫鏌ュ彞瀛愮骇鍒閿欑粨鏋 涓ユ牸锛氭墍鏈夋鏌ュ嚭鐨勪綅缃槸瀛樺湪閿欒鐨勪綅缃
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num and original_wrong_num != 0 and corrected_num == true_detected_char_in_sentence)
        else:#涓嶄弗鏍兼鏌ワ細 鍙ュ瓙瀛樺湪閿欒 涓旀湁妫鏌ョ棔杩
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)
        # if corrected_num != 0 and original_wrong_num != 0:
        if true_detected_flag:#妫閿欑粨鏋  鍙ュ瓙绾у埆妫閿 鍙鍙ュ瓙涓瓨鍦ㄩ敊璇 涓斿垽鏂ā鍨嬫湁鏀瑰姩 鍒欒涓烘槸姝ｇ‘鐨勬鏌ュ嚭浜嗛敊璇殑鍙ュ瓙
            #瀛楄瘝绾у埆妫閿 鍙ュ瓙涓瓨鍦ㄩ敊璇 涓旀墍鏈夐敊璇殑鍦版柟鍧囪姝ｇ‘鐨勬鏌ュ嚭鏉
            true_detected_sent += 1
        if correct == predict:#婊¤冻妯″瀷杈撳嚭鍙ュ瓙 绛変簬 姝ｇ‘鍙ュ瓙
            accurate_corrected_sent += 1 #杈撳嚭鍙ュ瓙绛変簬correct鐨勬暟鐩+1锛堝氨鏄纭殑鍙ュ瓙涓嶅仛鏀瑰彉+閿欒鐨勫彞瀛愮籂姝ｆ纭 Tp + Tn锛
        if correct == predict or true_detected_flag: #婊¤冻妯″瀷杈撳嚭鍙ュ瓙 绛変簬 姝ｇ‘鍙ュ瓙 鎴栬呮閿欑粨鏋滄纭
            accurate_detected_sent += 1 #姝ｇ‘妫鏌ュ嚭鐨勫彞瀛愭暟鐩+1

    print("corretion:")
    print(f'char_p={true_corrected_char}/{corrected_char}') #姝ｇ‘绾犳鐨刢har鏁扮洰/鎵鏈夌籂姝ｇ殑char鏁扮洰
    print(f'char_r={true_corrected_char}/{wrong_char}')#姝ｇ‘绾犳鐨刢har鏁扮洰/鎵鏈夊瓨鍦ㄩ敊璇殑char鏁扮洰
    print(f'sent_p={true_corrected_sent}/{corrected_sent}')#姝ｇ‘绾犳鐨勫彞瀛愭暟鐩/鎵鏈夌籂姝ｇ殑鍙ュ瓙鏁扮洰
    print(f'sent_r={true_corrected_sent}/{wrong_sent}') #姝ｇ‘绾犳鐨勫彞瀛愭暟鐩/鎵鏈夊瓨鍦ㄩ敊璇殑鍙ュ瓙鏁扮洰
    print(f'(Tp+Tn)/all_sents sent_a={accurate_corrected_sent}/{all_sent}') #锛圱p +Tn锛/ all_sents
    print("detection:")
    print(f'char_p={true_detected_char}/{corrected_char}')#姝ｇ‘妫鏌ュ嚭鐨刢har鏁扮洰 / 鎵鏈夋鏌ュ嚭鐨刢har鏁扮洰
    print(f'char_r={true_detected_char}/{wrong_char}')#姝ｇ‘妫鏌ュ嚭鐨刢har鏁扮洰 / 鎵鏈夊瓨鍦ㄩ敊璇殑char鏁扮洰
    print(f'sent_p={true_detected_sent}/{corrected_sent}')#姝ｇ‘妫鏌ュ嚭閿欒鐨勫彞瀛愭暟鐩 / 鎵鏈夌籂姝ｇ殑鍙ュ瓙鏁扮洰
    print(f'sent_r={true_detected_sent}/{wrong_sent}')#姝ｇ‘妫鏌ュ嚭閿欒鐨勫彞瀛愭暟鐩 / 鎵鏈夊瓨鍦ㄩ敊璇殑鍙ュ瓙鏁扮洰
    print(f'(Tp+Tn)/all_sents sent_a={accurate_detected_sent}/{all_sent}')#锛圱p + Tn )/ all_sentences

    #灏嗘ā鍨婥SC鍚庣殑缁撴灉鍙堝啓鍏ュ埌out_name.json鏂囦欢涓 鏂逛究鍚庣画璇曢獙瑙傚療
    w = open(f'{out_name}.json', 'w', encoding='utf-8')
    w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()


def parse_args():
    usage = '\n1. You can spell check several sentences by:\n' \
            'python faspell.py 鎵悧鍏虫敞涔堜紬鍙 鍙楀鑹哄叏缃戦鎾 -m s\n' \
            '\n' \
            '2. You can spell check a file by:\n' \
            'python faspell.py -m f -f /path/to/your/file\n' \
            '\n' \
            '3. If you want to do experiments, use mode e:\n' \
            ' (Note that experiments will be done as configured in faspell_configs.json)\n' \
            'python faspell.py -m e\n' \
            '\n' \
            '4. You can train filters under mode e by:\n' \
            'python faspell.py -m e -t\n' \
            '\n' \
            '5. to train filters on difference under mode e by:\n' \
            'python faspell.py -m e -t -d\n' \
            '\n'
    parser = argparse.ArgumentParser(
        description='A script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)

    parser.add_argument('multiargs', nargs='*', type=str, default=None,
                        help='sentences to be spell checked')
    parser.add_argument('--mode', '-m', type=str, choices=['s', 'f', 'e'], default='s',
                        help='select the mode of using FASPell:\n'
                             ' s for spell checking sentences as args in command line,\n'
                             ' f for spell checking sentences in a file,\n'
                             ' e for doing experiments on FASPell')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='under mode f, a file to be spell checked should be provided here.')
    parser.add_argument('--difference', '-d', action="store_true", default=False,
                        help='train on difference')
    parser.add_argument('--train', '-t', action="store_true", default=False,
                        help='True=to train FASPell with confidence-similarity graphs, etc.'
                             'False=to use FASPell in production')

    args = parser.parse_args()
    return args


def main():
    spell_checker = SpellChecker()
    args = parse_args()
    if args.mode == 's':  # command line mode
        try:

            assert args.multiargs is not None
            assert not args.train

            logging.basicConfig(level=logging.INFO)
            repeat_non_test(args.multiargs, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Sentences to be spell checked cannot be none.")

    elif args.mode == 'f':  # file mode
        try:
            assert args.file is not None
            sentences = []
            for sentence in open(args.file, 'r', encoding='utf-8'):
                sentences.append(sentence.strip())
            repeat_non_test(sentences, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Path to a txt file cannot be none.")

    elif args.mode == 'e':  # experiment mode
        
        if args.train:
            repeat_test(CONFIGS["exp_configs"]["training_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)
            # assert not CONFIGS["exp_configs"]["union_of_sims"]  # union of sims is a strategy used only in testing
            name = f'difference_{int(args.difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_0' #缁樺埗缁撴灉鍥句互鍙婅緭鍑虹浉搴斿瓧娈典俊鎭紝杩欓噷涔熸彁閱掓敞鎰忥細鍦ㄨ缁冩椂round鍙兘涓1锛屽洜涓烘澶勫彧鏄兘瀵圭涓杞畆ound鐨勭粨鏋滆繘琛岀粯鍒
            plot.plot(f'{name}.json',
                      f'{name}.txt',
                      store_plots=CONFIGS["exp_configs"]["store_plots"],
                      plots_to_latex=CONFIGS["exp_configs"]["store_latex"])
        else:
            repeat_test(CONFIGS["exp_configs"]["testing_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)


if __name__ == '__main__':
    main()

    '''
    Faspell鐨勭籂閿欑瓥锛氱粰瀹氫竴涓彞瀛 abcdef 鎸囧畾绾犳鐨剅ank锛堝欓夐泦澶у皬锛 round锛堥噸澶嶇籂姝ｆ鏁帮級
    
    瀵逛簬a杩涘叆MLM鍚 閫夋嫨鍓峳ank涓猼oken锛堟寜姒傜巼锛変綔涓哄欓夐泦 锛坅1,a2,a3锛
    鍦ㄨrank鍐呭惊鐜紝渚濇姣旇緝a鍜宎1锛宎2锛宎3鐨勫 濡傛灉a = a1缁х画姣旇緝鍚庨潰鐨勫欓夛紝濡傛灉a锛=a1 鍒欏a鍜宎1杩涜鍒ゆ柇
    鏍规嵁a1涓巃鐨剆imilarity鍜宑onfidence 杩涘叆涔嬪墠璁粌濂界殑鍒嗙被鍣紝濡傛灉鍒ゅ畾涓烘槸涓涓敊璇紝鍒欏皢a鏀逛负a1 鍚﹀垯缁х画姣旇緝鍚庨潰
    鐨勫欓夈傚綋姣旇緝瀹屽欓夐泦浠嶆病鎵惧埌閿欒鏃讹紝鍒欏垽瀹歛涓烘纭紝涓嶅仛淇銆
    鎵浠ュ浜嶧ASPell鑰岃█锛
    1.MLM鑳藉皢瓒婂鐨勯敊璇慨姝ｇ‘瀹氬湪鍓峳ank涓互鍙婁笉灏嗘纭殑鏀逛负閿欒鐨勮兘鍔涘緢閲嶈锛岃繖鐩存帴鍐冲畾浜嗘ā鍨嬬殑涓婄嚎锛
    2.瀵逛簬CSD鐨勫垎绫诲櫒涔熷紓甯搁噸瑕侊紝濡備綍纭繚妯″瀷鍦╮ank涓壘鍒版纭殑瀛楋紝杩囨护涓嶅繀瑕佺殑淇敼浠ュ強涓嶆纭殑淇敼锛
    瀵逛簬鎻愰珮榄旀х殑鎬ц兘璧峰埌浜嗗叧閿綔鐢ㄣ
    
    '''
