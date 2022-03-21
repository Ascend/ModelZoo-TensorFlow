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
import json
import matplotlib.pyplot as plt
import os


def plot(json_fname, results_fname, store_plots='', plots_to_latex=''):
    name = '.'.join(json_fname.split('.')[:-1])

    data = json.loads(open(json_fname, 'r', encoding='utf-8').read())#瀵规墍鏈夊彞瀛恈SC鍚庨敊璇殑缁撴灉 results = [[鍙ュ瓙1鐨剅esult],[鍙ュ瓙2鐨剅esult]锛...]
    fi = open(results_fname, 'r', encoding='utf-8')#瀵规墍鏈夊彞瀛恈sc鍚庣殑缁撴灉 line = origin_num, wrong_sent, correct_sent, predict_sent, num

    # data for a confidence-similarity graph
    truely_detected_and_truely_corrected = [[], []]
    truely_detected_and_falsely_corrected = [[], []]
    falsely_detected = [[], []]

    count_of_absence_of_correct_chars = [0, 0]

    w3 = open(f'{name}_falsely_detected.txt', 'w', encoding='utf-8')
    w4 = open(f'{name}_falsely_corrected.txt', 'w', encoding='utf-8')

    for line, entry in zip(fi, data):
        origin_num, wrong_sent, correct_sent, predict_sent, num = line.strip().split('\t')
        pos_to_error = dict([(e["error_position"], e) for e in entry["errors"]])
        for pos, (w, c, p) in enumerate(zip(wrong_sent, correct_sent, predict_sent)):
            if w != c and w != p:#姝ｇ‘妫鏌
                e = pos_to_error[pos]
                assert e["corrected_to"] == p
                if c != p:#姝ｇ‘妫鏌ラ敊璇籂姝
                    candidatas = dict(sorted(list(e["candidates"].items()), reverse=True, key=lambda it: it[1])[:5])
                    absent = 'no'
                    if c not in candidatas:#鏌ョ湅姝ｇ‘瀛楃鏄惁鍑虹幇鍦ㄤ簡candidatas涓
                        count_of_absence_of_correct_chars[0] += 1
                        absent = 'yes'
                    truely_detected_and_falsely_corrected[0].append(e["confidence"])
                    truely_detected_and_falsely_corrected[1].append(e["similarity"])

                    w4.write('\t'.join([wrong_sent,
                                        f'pos={pos}',
                                        f'w={w}',
                                        f'c={c}',
                                        f'p={p}',
                                        f'sim={e["similarity"]}',
                                        f'absent={absent}']) + '\n')
                else:#姝ｇ‘妫鏌ヤ笖姝ｇ‘绾犳
                    truely_detected_and_truely_corrected[0].append(e["confidence"])
                    truely_detected_and_truely_corrected[1].append(e["similarity"])

            elif w == c and w != p:#閿欒妫鏌 鏈韩姝ｇ‘鍗磋鍒ゅ畾涓洪敊璇
                e = pos_to_error[pos]
                candidates = dict(sorted(list(e["candidates"].items()), reverse=True, key=lambda it: it[1])[:5])
                absent = 'no'
                if c not in candidates:#鏌ョ湅姝ｇ‘瀛楃鏄惁鍑虹幇鍦ㄤ簡candidatas涓
                    count_of_absence_of_correct_chars[1] += 1
                    absent = 'yes'

                falsely_detected[0].append(e["confidence"])
                falsely_detected[1].append(e["similarity"])

                w3.write('\t'.join([wrong_sent,
                                    f'pos={pos}',
                                    f'w={w}',
                                    f'c={c}',
                                    f'p={p}',
                                    f'sim={e["similarity"]}',
                                    f'absent={absent}']) + '\n')

            elif w!=c and w==p:#鏈韩鏄敊璇殑 妯″瀷娌℃湁妫鏌ュ嚭鏉 CSC鍦ㄧ敾鍥剧殑鏃跺 鏈冭檻杩欎竴绫荤偣
                pass
                #print('==' * 20)
                #print(wrong_sent)
                #(w)
                #print(correct_sent)
                #print(predict_sent)
                #print('==' * 20)
    # print statistics
    print(f'In {len(truely_detected_and_falsely_corrected[0])} falsely corrected characters,'
          f' {count_of_absence_of_correct_chars[0]} are because of absent correct candidates.')
    print(f'In {len(falsely_detected[0])} falsely detected characters,'
          f' {count_of_absence_of_correct_chars[1]} are because of absent correct candidates.')

    #缁樺浘
    plt.plot(truely_detected_and_truely_corrected[0], truely_detected_and_truely_corrected[1], 'ro',
             truely_detected_and_falsely_corrected[0], truely_detected_and_falsely_corrected[1], 'bo',
             falsely_detected[0], falsely_detected[1], 'x')
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.show()

    #鏄惁淇濆瓨鍥句腑姣忎釜鐐圭殑璇︾粏淇℃伅
    if plots_to_latex:
        produce_latex(truely_detected_and_truely_corrected,
                      truely_detected_and_falsely_corrected,
                      falsely_detected, os.path.join(plots_to_latex, f'{name}_latex.txt'))
    if store_plots:#淇濆瓨鍥剧墖 涓ゅ紶 涓寮犳槸鍩虹鍥 鍙︿竴寮犳槸鏀惧ぇ鍩虹鍥句腑鐨勫彸涓婅閮ㄥ垎
        # plt.savefig(os.path.join(store_plots, f'{name}.png'))
        axes = plt.gca()
        # axes.set_xlim([0.95,1])
        # axes.set_ylim([0.0,0.3])
        plt.savefig(os.path.join(store_plots, f'{name}.png'))
        axes.set_xlim([0.95,1])
        axes.set_ylim([0.0,0.6])
        plt.savefig(os.path.join(store_plots, f'{name}2.png'))
        # plt.pause(0.0001)
        # plt.clf()


def produce_latex(truely_detected_and_truely_corrected, truely_detected_and_falsely_corrected, falsely_detected, path):
    #灏嗙粯鍒剁偣鐨勪俊鎭繘琛屼繚瀛
    f = open(path, 'w')
    for a_x, a_y in zip(truely_detected_and_truely_corrected[0], truely_detected_and_truely_corrected[1]):
        f.write(f'({a_x},{a_y})[a]')
    for b_x, b_y in zip(truely_detected_and_falsely_corrected[0], truely_detected_and_falsely_corrected[1]):
        f.write(f'({b_x},{b_y})[b]')
    for c_x, c_y in zip(falsely_detected[0], falsely_detected[1]):
        f.write(f'({c_x},{c_y})[c]')

    f.close()

