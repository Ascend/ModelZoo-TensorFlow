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
import numpy as np
import pickle
import os
import argparse

def cut_line(sentence):
    # 瀵瑰彞瀛愯繘琛屽垏鍓诧紝濡傛灉鍙ュ瓙涓棿瀛樺湪涓篬'銆', '锛', '锛', '锛']鑰屼笉鏄痆'銆', '鈥', '鈥']鐨勫彞瀛
    # 鍒欏皢璇ユ爣鐐圭鍙峰強涔嬪墠鐨勫唴瀹瑰仛涓轰竴涓崟鐙殑鍙ュ瓙銆
    # eg S1= '鎴戠湡鐨勫緢鍠滄浣狅紒浣犵煡閬撳悧锛' cut -> '鎴戠湡鐨勫緢鍠滄浣狅紒'  &  '浣犵煡閬撳悧锛'
    # eg S2= '鎴戠湡鐨勫緢鍠滄鈥滃ぉ浣库濅綘鐭ラ亾鍚楋紵' cut -> '鎴戠湡鐨勫緢鍠滄鈥滃ぉ浣库濅綘鐭ラ亾鍚楋紵'
    #涓嶈繃闇瑕佹敞鎰忕殑鏄紝寰楃泭浜巠ield浣滀负鍏抽敭瀛楋紝璋冪敤璇ュ嚱鏁扮殑缁撴灉搴旇浣跨敤for寰幆閬嶅巻


    sent = ''
    delimiter = ['銆', '锛', '锛', '锛']
    for i, c in enumerate(sentence):
        sent += c
        if ((c in delimiter) and (sentence[min(len(sentence)-1, i + 1)] not in ['銆', '鈥', '鈥'])) or i == len(sentence)-1:
            yield sent
            sent = ''
def cut_line2(sentence):
    #瀵瑰惈鏈夋爣鐐圭鍙穂,]鐨勫彞瀛愯繘琛岃繃婊わ紝涓昏鏄拡瀵归偅浜涢暱搴﹁繃闀跨殑鍙ュ瓙杩涜cut锛屽彧鏈夋弧瓒充互涓嬫潯浠剁殑鍙ュ瓙鎵嶈兘琚夋嫨
    #浠庡乏寰鍙崇殑姣忎竴涓猍,] 鍙瀛樺湪涓涓槸鍙ュ瓙鏈熬鎴栬呮弧瓒宠[,]鍚庨潰鐨6涓瓧绗︽病鏈塠,]涓
    #鍒板彞瀛愬ご鐨勯暱搴﹀ぇ浜20锛屽垯杩斿洖璇,]鍓嶇殑鍐呭锛岀粨灏剧鍙蜂负[銆俔
    sent = ''
    for i, c in enumerate(sentence):
        sent += c
        if c == '锛':
            flag = True
            for j in range(i+1, min(len(sentence)-1, i+6)):
                if sentence[j] == '锛' or j == len(sentence)-1:
                    # 瀹為檯涓婏紝杩欎釜鍦版柟鏃犺鎬庝箞鏍 閮戒笉鑳借揪鍒癹 == len(sentence)-1鐨勬潯浠
                    flag = False

            if (flag and len(sent) > 20) or i == len(sentence)-1:
                yield sent[:-1] + '銆'
                sent = ''



def make_docs(wrong, correct):
    #鎸夌収涓婅堪涓や釜鍙ュ瓙鐨刢ut鏂规硶灏嗚缁冩暟鎹殑wrong鍜宑orrect鍙ュ瓙杩涜澶勭悊锛屽苟鍐欏叆鍒板搴旂殑鏂囦欢涓
    #鎴戣寰楄繖涓柟娉曠殑閫昏緫鍐欑殑鏈夌偣娣蜂贡
    w_res = []
    #瀵规弧瓒充笅闈㈡潯浠剁殑鍙ュ瓙杩涜鍒囧垎
    if ('銆' in wrong[:-1]) or ('锛' in wrong[:-1]) or ('锛' in wrong[:-1]) or ('锛' in wrong[:-1]):
        for w_sent in cut_line(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    elif len(wrong) > 100:#瀵瑰彞瀛愰暱搴﹀ぇ浜100鐨勫彞瀛愬垏鍒
        for w_sent in cut_line2(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    else:#涓嶅仛鍒囧垎澶勭悊
        w_res.append(wrong + '\n')
        # wrong_file.write(wrong + '\n')

    # wrong_file.write('\n')
    c_res = []
    if ('銆' in correct[:-1]) or ('锛' in correct[:-1]) or ('锛' in correct[:-1]) or ('锛' in correct[:-1]):
        for c_sent in cut_line(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    elif len(wrong) > 100:
        for c_sent in cut_line2(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    else:
        c_res.append(correct + '\n')
        # correct_file.write(correct + '\n')

    #鎸夌収涓婅堪鏂规硶澶勭悊鍚庣殑鍙ュ瓙缁撴灉妫鏌ワ紝濡傛灉涓や釜list鐨勯暱搴︿笉鐩哥瓑锛屽垯涓嶅仛涓婅堪澶勭悊锛岀洿鎺ュ皢鍘熷彞鏀惧叆list
    if len(w_res) != len(c_res):
        w_res = [wrong + '\n']
        c_res = [correct + '\n']

    #濡傛灉wrong鍜宑orrect瀵瑰簲鐨勫彞瀛愰暱搴︿笉鐩哥瓑锛宔xit()
    for w_r, c_r in zip(w_res, c_res):
        if not len(w_r.strip()) == len(c_r.strip()):
            print(w_r)
            print(len(w_r.strip()))
            print(c_r)
            print(len(c_r.strip()))
            exit()
    #妫鏌ヨ繃鍚庡皢澶勭悊鐨勬暟鎹啓鍏ュ埌鏂囦欢涓
    for l in w_res:
        wrong_file.write(l)
    wrong_file.write('\n')

    for l in c_res:
        correct_file.write(l)
    correct_file.write('\n')


def main(fname, output_dir):
    confusions = {}

    for line in open(fname, 'r', encoding='utf-8'):
        #璇诲叆棰勮缁冩ā鍨嬩娇鐢ㄧ殑鏁版嵁 eg锛 1	鎶婅繖涓瀛愮摚杈	鎶婅繖涓瀛愮摚鐡  瀵瑰簲鐨勬槸锛氶敊璇暟鐩 閿欒鍙ュ瓙 姝ｇ‘鍙ュ瓙
        #print(line)
        #print(line.strip().split('\t'))
        num, wrong, correct = line.strip().split('\t')
        wrong = wrong.strip()
        correct = correct.strip()
        for w, c in zip(wrong, correct):
            #濡傛灉涓や釜鍙ュ瓙瀵瑰簲瀵瑰簲浣嶇疆鐨勫瓧绗︿笉涓鑷
            if w!=c:
                #灏唚+c娣诲姞鍒癱onfusion瀛楀吀涓紝 鏋勯犲洶鎯戞浛鎹㈤泦
                #eg锛歸rong: abcd correct:cbcd    confusion={'ac':0} ac鏄竴涓洶鎯戞浛鎹㈠ a琚浛鎹㈡垚浜哻
                if w + c not in confusions:
                    confusions[w + c] = 0
                confusions[w + c] += 1
        # if len(wrong) != len(correct):
        #     print(wrong)
        #     print(correct)
        #     exit()

        #鍒ゆ柇wrong 鍜 correct 鍙ュ瓙鐨勯暱搴︽槸鍚﹀绛
        assert len(wrong) == len(correct)
        #鍦ㄨ缁剋rong鍜宑orrect瀵逛腑瀛樺湪鐨勯敊璇鏁
        num = int(num)
        #澶勭悊wrong鍜宑orrect鍙ュ瓙瀵
        make_docs(wrong, correct)
        #瀵逛簬wrong鍜宑orrect瀛樺湪涓嶄竴鑷寸殑鍙ュ瓙瀵癸紝浣跨敤correct-correct鏋勫缓璐熸牱鏈
        if wrong != correct:
            make_docs(correct, correct)

        #缁熻wrong鍜宑orrect涓 瀵瑰簲浣嶇疆涓嶄竴鑷寸殑瀛楃鎵瀵瑰簲鐨刬ndex
        #wrong: abcd correct:cbcd         poses=[0]  瀵瑰簲涓嶄竴鑷寸殑涓嬫爣涓0
        poses = [pos for pos, (w, c) in enumerate(zip(wrong, correct)) if w != c]
        num = len(poses) #姝ゅ杩欎竴鏉¤鍙ユ病鏈夋剰涔夛紙涓嬮潰鐨勫垽鏂細if len(poses) != num灏变笉搴旇瀛樺湪浜嗭級 鎵浠ユ垜鍦ㄨ繍琛岀殑鏃跺欐槸娉ㄩ噴鎺夎鏉¤鍙ョ殑



        #瀵逛簬wrong correct瀛樺湪涓嶄竴鑷寸殑瀛楃鏁板ぇ浜庣瓑浜2鐨勶紝浠ユ涓哄熀纭缁х画鏋勫缓閿欒
        #eg: wrong: abcde correct:fghde there num=3
        # -> wrong1 corect wrong1涓轰笂杩颁笁涓敊璇腑鐨勪换鎰忎竴涓
        # -> wrong2 corect wrong2涓轰笂杩颁笁涓敊璇腑鐨勪换鎰忎簩涓
        if num >= 2:
            #鍒ゆ柇 濡傛灉璁＄畻鍑簑rong鍜宑orrect涓嶄竴鑷寸殑瀛楃鏁 涓嶇瓑浜 num 閫鍑
            if len(poses) != num:
                print(wrong)
                print(correct)
                exit()
            assert len(poses) == num
            for i in range(1, num):
                selected_poses = [poses[k] for k in np.random.choice(num, i, replace=False)]
                fake_wrong = list(wrong)
                for p in selected_poses:
                    fake_wrong[p] = correct[p]

                fake_wrong = ''.join(fake_wrong)
                assert len(fake_wrong) == len(correct)
                assert fake_wrong != correct
                make_docs(fake_wrong, correct)

    # take the top frequency of confusions about the each character.

    #鍦ㄨ缁冮泦涓 涓涓瓧琚浛鎹负鍝竴涓瓧鐨勯鐜囨渶楂橈紝鍒欓夋嫨杩欎釜鏇挎崲瀵瑰嚭鐜扮殑娆℃暟浣滀负璇ュ瓧鐨勮鏇挎崲棰戠巼
    top_confusions = {}
    for k in confusions:
        if k[0] not in top_confusions:
            top_confusions[k[0]] = confusions[k]
        elif top_confusions[k[0]] < confusions[k]:
            top_confusions[k[0]] = confusions[k]

    #鎸夌収棰戠巼杩涜鎺掑簭锛屽苟鍙彇key鍊
    confusions_top = sorted(list(top_confusions.keys()), key=lambda x: top_confusions[x], reverse=True)

    #瀵逛簬閭ｄ簺鍦ㄦ煇浜涘彞瀛愪腑琚涓烘槸閿欏瓧鐨勫瓧 鍦ㄥ彟澶栧彞瀛愪腑鍗村張鏄纭瓧 鐨勫瓧鐨勬寜棰戠巼缁熻
    correct_count = {}
    for line_c, line_w in zip(open(os.path.join(args.output, 'correct.txt'), 'r', encoding='utf-8'), open(os.path.join(args.output, 'wrong.txt'), 'r', encoding='utf-8')):
        if line_c.strip():
            wrong, correct = line_w.strip(), line_c.strip()#strip()榛樿绉婚櫎绌烘牸鍜屾崲琛 鎵浠ヤ笉闇瑕佹樉绀烘寚瀹歴trip('\n')锛屾墍浠ヤ笅闈㈢揣璺熺殑涓ゆ潯璇彞瀹為檯涔熶笉闇瑕
            wrong = wrong.strip()
            correct = correct.strip()
            for w, c in zip(wrong, correct):
                if w==c and w in top_confusions:
                    if w not in correct_count:
                        correct_count[w] = 0
                    correct_count[w] += 1
    #瀵逛簬涓涓瓧 鏈夋椂浼氭槸閿欏瓧 鏈夋椂浼氭槸姝ｇ‘瀛 鐨勬瘮渚嬭绠 p = min(琚浛鎹㈢殑娆℃暟 / 鏈鏇挎崲鐨勬鏁,1.0) 涔熸槸鍚庨潰鐨凪ASK姒傜巼
    proportions = {}
    for k in correct_count:
        assert correct_count[k] != 0
        proportions[k] = min(top_confusions[k] / correct_count[k], 1.0)

    print('confusion statistics:')

    for i in range(min(len(confusions_top), 20)):
        if confusions_top[i] in correct_count:
            correct_occurs = correct_count[confusions_top[i]]
            proportions_num = proportions[confusions_top[i]]
        else:
            correct_occurs = 0
            proportions_num = 'NaN'
        print(f'most frequent confusion pair for {confusions_top[i]} occurs {top_confusions[confusions_top[i]]} times,'
              f' correct ones occur {correct_occurs} times, mask probability should be {proportions_num}')

    pickle.dump(proportions, open(os.path.join(args.output, 'mask_probability.sav'), 'wb'))
    # print('top confusions:')
    # for i in range(20):
    #     print(f'{top_confusions[i]} occurs {confusions[confusions_top[i]]} times')

'''
瀵逛簬鍑虹幇鐨勫嚑涓瓧鍏哥殑璇︾粏瑙ｉ噴锛
confusions 鐢ㄦ潵璁板綍鍥版儜鏇挎崲闆  key锛氶敊璇彞瀛愪腑鐨勪竴涓瓧琚浛鎹㈡垚姝ｇ‘鍙ュ瓙涓殑鍙﹀涓涓瓧锛堢浉褰撲簬琚籂姝ｇ殑鎰忔濓級 value锛氬嚭鐜伴鐜
top_confusions 涓涓瓧鑷繁瀵瑰簲鐨勫洶鎯戞浛鎹㈠涓 閫夊彇棰戠巼鏈楂樼殑浣滀负鑷繁琚浛鎹紙鍑洪敊锛夌殑棰戠巼
            key锛氳鏇挎崲鐨勫瓧 value锛氭浛鎹㈤鐜
confusions_top 瀵箃op_confusions鐨勬帓搴忥紙浠庨珮鍒颁綆锛 骞跺彧淇濈暀key鍊 鏄竴涓猯ist
correct_count 瀵逛簬閭ｄ簺鍦ㄦ煇浜涘彞瀛愪腑鏄敊瀛 浣嗗湪鍙﹀鍙ュ瓙涓張鏄纭殑瀛楃殑缁熻
            key:鏃細琚涓烘槸閿欏瓧鍙堜細琚涓烘槸姝ｇ‘瀛楃殑瀛 value锛氳璁や负鏄纭殑瀛楃殑棰戠巼

proportions key:鏃細琚涓烘槸閿欏瓧鍙堜細琚涓烘槸姝ｇ‘瀛楃殑瀛 value锛歱 = min(琚浛鎹㈢殑娆℃暟 / 鏈鏇挎崲鐨勬鏁,1.0)

eg:
瀵逛簬瀛    鈥樺ぉ鈥 鍦ㄨ缁冮泦wrong涓嚭鐜颁簡4娆 瀵瑰簲correct涓垎鍒槸 鈥樼敯鈥欌樼敯鈥欌樺～鈥 鈥樺ぉ鈥
 confusions = {'澶╃敯':2, '澶╁～':1}    
 top_confusions = {'澶':2}   
 confusions_top = ['澶']
 correct_count = {'澶':1}
 proportions = {'澶':1}
'''

# main()
def parse_args():
    #浣跨敤鍛戒护浠嬬粛 璇y鏂囦欢鏄敤鏉ユ牴鎹師濮嬫枃浠跺垱寤虹埍瀵瑰簲鐨勪笌璁粌闆嗗悎mask姒傜巼鍊
    usage = '\n1. create wrong.txt, correct.txt and mask_probability.sav by:\n' \
            'python create_data.py -f /path/to/train.txt\n' \
            '\n2. specify output dir by:\n' \
            'python create_data.py -f /path/to/train.txt -o /path/to/dir/\n' \
            '\n' 
    parser = argparse.ArgumentParser(
        description='A module for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)
    #鍘熷杈撳叆鏂囦欢
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='original training data.')
    #鎸囧畾杈撳嚭鏂囦欢鐨勮矾寰 榛樿涓哄綋鍓嶈矾寰
    parser.add_argument('--output', '-o', type=str, default='',
                        help='output a file a dir; default is current directory.')
    # parser.add_argument('--verbose', '-v', action="store_true", default=False,
    #                     help='to show details of spell checking sentences under mode s')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    correct_file = open(os.path.join(args.output,'correct.txt'), 'w', encoding='utf-8')
    wrong_file = open(os.path.join(args.output,'wrong.txt'), 'w', encoding='utf-8')
    main(args.file, args.output)

