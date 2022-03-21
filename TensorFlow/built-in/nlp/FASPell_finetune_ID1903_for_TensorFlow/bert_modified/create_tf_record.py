# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import collections
import random
import tokenization
import tensorflow as tf
import pickle

#鑾峰彇token鐨勭浉鍏矼ASK姒傜巼
MASK_PROB = pickle.load(open('mask_probability.sav', 'rb'))
WRONG_COUNT = dict([(k, 0) for k in MASK_PROB])
CORRECT_COUNT = dict([(k, 0) for k in MASK_PROB])

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

flags = tf.flags

FLAGS = flags.FLAGS
#瀹氫箟杈撳叆鏂囦欢
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
#瀹氫箟wrong杈撳叆鏂囦欢
flags.DEFINE_string("wrong_input_file", None,
                    "same as input_file except containing wrong characters.")
#瀹氫箟杈撳嚭鏂囦欢 涓鑸负tf_examples.tfrecord
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")
#瀹氫箟璇嶈〃鏂囦欢 涓鑸负bert涓殑vocab.txt
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
#瀹氫箟鏄惁杞皬鍐
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
#瀹氫箟鏈澶eq闀垮害
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
#瀹氫箟鍙娴嬬殑鏈澶ч暱搴 涔熷氨鏄渶澶歮ask20涓猼oken
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
#瀹氫箟闅忔満绉嶅瓙
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

#瀹氫箟閲嶅娆℃暟 榛樿10 涔熷氨鏄浜庝竴涓緭鍏 杩涜10娆￠噸澶嶇殑mask鎿嶄綔锛堟瘡娆ask鐨勪綅缃細涓嶄竴鏍凤級
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")
#瀹氫箟mask姒傜巼
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
#瀹氫箟鍒涘缓seq鏃舵瘮鏈澶ч暱搴︾煭鐨勬鐜
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

#瀹氫箟璁粌瀹炰緥
class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens #璁粌瀹炰緥瀵瑰簲鐨則okens ([CLS] 鎴 寰 ...)
        self.segment_ids = segment_ids #璁粌瀹炰緥瀵瑰簲鐨剆egment_ids (00001111...)
        self.is_random_next = is_random_next #璁粌瀹炰緥瀵瑰簲鐨勪袱涓彞瀛愪箣闂存槸鍚︽瀯鎴愪笂涓嬫枃鍏崇郴
        self.masked_lm_positions = masked_lm_positions #璁粌瀹炰緥瀵瑰簲鐨刴ask浣嶇疆  (001001...)
        self.masked_lm_labels = masked_lm_labels #璁粌瀹炰緥瀵瑰簲琚玬ask鎺夌殑鍘焧oken锛屼篃灏辨槸lebel

    #灏嗚缁冨疄渚嬪瓧绗︿覆鍖 鏂逛究杈撳嚭灞曠ず
    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

#灏嗗崟涓鐨勮缁冨疄渚嬭浆涓篹xample骞跺啓鍏F
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    #鏍规嵁浼犲叆鐨勫啓鍏F鏂囦欢鍒楄〃锛坥utput_files 澶氫釜锛夋瘡涓涓猅F鏂囦欢鍒涢犱竴涓啓鍏riter
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        #鑾峰彇姣忎竴涓疄渚嬩腑鐨勭浉鍏冲瓧娈典俊鎭
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length #鏂畾input_ids鏄惁灏忎簬绛変簬鏈澶ц緭鍏equence闀垮害

        while len(input_ids) < max_seq_length:#瀵逛簬灏忎簬鐨勯儴鍒 鐢0琛ラ綈
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length #琛ラ綈鍚庡啀涓娆℃柇瀹
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions) #鑾峰彇瀹炰緥涓mask鐨勪綅缃
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels) #鑾峰彇瀹炰緥涓mask鎺夌殑浣嶇疆瀵瑰簲鐨刲abel
        masked_lm_weights = [1.0] * len(masked_lm_ids) #瀹氫箟琚玬ask鎺夌殑lm_weights

        #瀵逛簬姣忎竴涓簭鍒楁渶澶氬彲浠ask鎺夌殑闀垮害 骞剁敤0杩涜琛ラ綈
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        #纭畾涓や釜鍙ュ瓙涔嬮棿鏄惁澶熼暱涓婁笅鏂囧叧绯
        next_sentence_label = 1 if instance.is_random_next else 0

        #寤虹珛feature鍜宔xample鐨勬槧灏 瀹炵幇灏嗚缁冨疄渚嬪簭鍒楀寲鍐欏叆tf鏂囦欢涓
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        #鍐欏叆鍒扮浉搴旂殑TF鏂囦欢涓
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        #璁板綍鍐欏叆鍒癟f涓殑鎬籩xample
        total_written += 1

        #杈撳嚭鍐欏叆鐨勫墠20鏉′俊鎭
        if inst_index < 20:
            # pass
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    #杈撳嚭鍦∕ASK鍚 鍦ㄦ墍鏈夊疄渚嬩腑 姣忎釜token锛坱oken鏉ヨ嚜mask_probability锛夎mask鐨勬鏁帮紙鍙緭鍑哄墠10涓級
    for k in sorted(list(WRONG_COUNT.keys()), key=lambda x: WRONG_COUNT[x] + CORRECT_COUNT[x], reverse=True)[:10]:
        print(f'correct {k} is masked {WRONG_COUNT[k] + CORRECT_COUNT[k]} times in all instances')

    #鍏抽棴writer
    for writer in writers:
        writer.close()
    #鎵撳嵃鎬荤殑examples鏁扮洰
    tf.logging.info("Wrote %d total instances", total_written)

#鍒涘缓feature鍜宔xample鏄犲皠鏄殑feature瀹氫箟 鍥犱负鎵鏈夌殑瀛楁閮芥槸锛坕d锛塱nt64浣 鎵浠ヤ娇鐢╰f.train.Int64List
def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

#鍒涘缓feature鍜宔xample鏄犲皠鏄殑feature瀹氫箟 鍥犱负鎵鏈夌殑瀛楁閮芥槸锛坕d锛塱nt64浣 鎵浠ヤ娇鐢╰f.train.FloatList
def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


#鏍规嵁杈撳叆鐨刢orrect wrong 鍜 tokenizer 鏋勯犺缁冨疄渚
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, wrong_input_files):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]] #璁板綍鎵鏈夌殑correct lines 閲囩敤鍒楄〃涓殑鍒楄〃 褰㈠紡杩涜璁板綍

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                #閫愯璇诲彇杈撳叆鏂囦欢
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:#绋冲綋璇诲彇瀹屾瘯 閫鍑哄惊鐜
                    break
                line = line.strip()#鍘婚櫎鏈熬鐨勬崲琛岀

                # Empty lines are used as document delimiters
                if not line:#璇诲彇鍒扮┖琛
                    all_documents.append([]) #涔嬫墍浠ュ湪鏋勫缓correct鍜寃rong鏃跺欓渶瑕佹湁绌鸿鍋氶棿闅 灏辨槸涓轰簡鍦ㄨ繖閲岃繘琛屾暟鎹鐞 涔熷氨鏄墍璇寸殑鏂囨。闂撮殧
                tokens = tokenizer.tokenize(line) #瀵筶ine杩涜tokenize
                if tokens:#灏唗okenize鍚庣殑line鍔犲叆鍒癮ll_docuents鏈鍚庝竴涓瓙鍏冪礌涓
                    all_documents[-1].append(tokens)
                #result锛歛ll_documents = [[[t1,t2..]],[[t1,...]],[[t1,t2,t3...]],[[t1,...]],...,[[t1,t2...]]]
    # Remove empty documents
    all_documents = [x for x in all_documents if x]#鍘绘帀绌虹殑瀛愬厓绱狅紙list锛

    # rng.shuffle(all_documents)

    all_wrong_documents = [[]]#璁板綍鎵鏈夌殑wrong lines 閲囩敤鍒楄〃涓殑鍒楄〃 褰㈠紡杩涜璁板綍

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for wrong_input_file in wrong_input_files:
        with tf.gfile.GFile(wrong_input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_wrong_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_wrong_documents[-1].append(tokens)

    # Remove empty documents
    all_wrong_documents = [x for x in all_wrong_documents if x]


    #瀵规牴鎹甤orrect 鍜 wrong 鏋勫缓鐨勬暟鎹繘琛屾柇瀹 浜岃呯殑闀垮害搴旇鐩哥瓑
    assert len(all_documents) == len(all_wrong_documents)

    # rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, all_wrong_documents))

    rng.shuffle(instances)#鏁版嵁鎵撲贡
    return instances



#鏋勫缓璁粌瀹炰緥
def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, all_wrong_documents, original=True):
    """Creates `TrainingInstance`s for a single document."""
    #鏍规嵁鎸囧畾涓嬫爣document_index鑾峰彇鎸囧畾鐨刢orrect鍜寃rong document
    document = all_documents[document_index]
    wrong_document = all_wrong_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    #鍥犱负瑕佸湪tokenize鍚 鎻掑叆涓変釜鐗规畩绗﹀彿[CLS], [SEP], [SEP]鎵浠ヨ兘鍏佽鐨勬渶澶okens鐨勯暱搴﹀簲璇ュ湪max_seq_length鍩虹涓-3
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:#鐢熸垚涓涓01闂寸殑float 濡傛灉灏忎簬short_seq_prob锛堟瘮鏈澶equence闀垮害灏忕殑姒傜巼锛 灏遍噸鏂拌瀹歵arget_seq_length鐨勯暱搴
        target_seq_length = rng.randint(2, max_num_tokens)#鐢熸垚2鍒癿ax_num_tokens涔嬮棿鐨勪竴涓猧nt鏁

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_wrong_chunck = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        wrong_segment = wrong_document[i]
        current_chunk.append(segment)
        current_wrong_chunck.append(wrong_segment)

        # assert len(segment) == len(wrong_segment)
        current_length += len(segment)
        try:
            assert len(segment) == len(wrong_segment)
        except:
            print(segment)
            print(wrong_segment)
            exit()
        # assert segment == wrong_segment

        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                wrong_tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    wrong_tokens_a.extend(current_wrong_chunck[j])

                assert len(tokens_a) == len(wrong_tokens_a)

                tokens_b = []
                wrong_tokens_b = []
                # Random next 闅忔満鐨勪笅鍙
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_wrong_document = all_wrong_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        wrong_tokens_b.extend(random_wrong_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next 瀹為檯鐨勪笅鍙
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                        wrong_tokens_b.extend(current_wrong_chunck[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng, wrong_tokens_a, wrong_tokens_b)
                # truncate_seq_pair(wrong_tokens_a, wrong_tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                wrong_tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                wrong_tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                for w_token in wrong_tokens_a:
                    wrong_tokens.append(w_token)
                    # segment_ids.append(0)

                tokens.append("[SEP]")
                wrong_tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                for w_token in wrong_tokens_b:
                    wrong_tokens.append(w_token)

                tokens.append("[SEP]")
                wrong_tokens.append("[SEP]")
                segment_ids.append(1)

                if tokens == wrong_tokens:#瀵逛簬姝ｇ‘鍙ュ瓙鐨凪ASK绛栫暐
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                else:#瀵逛簬閿欒鍙ュ瓙鐨凪ASK绛栫暐
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = create_masked_lm_predictions_for_wrong_sentences(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, wrong_tokens)
                    # print(tokens)
                    # print(wrong_tokens)
                    # print(masked_lm_positions)
                    # print(masked_lm_labels)
                    # print('\n')
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)

            current_chunk = []
            current_wrong_chunck = []
            current_length = 0
        i += 1
    '''
    鍦ㄦ瘡涓娆or寰幆涓 鍒涘缓涓涓猧nstances瀹炰緥 涓婅堪浠ｇ爜寰堥毦寮勬噦寰楅儴鍒嗘槸鍏充簬涓嬩竴鍙ラ娴 is_random_next = 1(涓嶆瀯鎴愪笂涓嬫枃鍏崇郴)0(鏋勬垚涓婁笅鏂囧叧绯)
    tokens = [CLS]s1[SEP]s2[SEP]
    segment_ids = [0]+[0]*len(s1)+[1]*len(s2)+[1]
    is_random_next = 0/1
    masked_lm_positions = 29, 582, 120,...
    masked_lm_labels = 213,415,312,...
    assert len(tokens) = len(segment_ids) = len(masked_lm_positions) = len(masked_lm_labels)
    '''
    return instances

#瀵逛簬姝ｇ‘鍙ュ瓙鐨凪ASK绛栫暐
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    #鏍规嵁浼犲叆鐨則okens鍜岀浉鍏充俊鎭繘琛宮ask
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)#鎵撲贡index 鍏锋湁椴佹鎬

    output_tokens = list(tokens)
    #max_predictions_per_seq 姣忎竴涓猻eq鏈澶歮ask鐨勬暟鐩 masked_lm_prob 姣忎竴涓猻eq鎺╃洊鐨刴ask姣斾緥锛0.15锛
    #纭畾鐩爣鎺╃洊鏁扮洰
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))


    masked_lms = [] #瀛樺偍琚玀ASK浣嶇疆鍜岀浉搴旂殑label
    covered_indexes = set()#瀛樺偍瀵逛竴涓猻eq瀹為檯鎺╃洊鐨刬ndex
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:#瓒呭嚭浜嗗噯璁告帺鐩栨垨鑰呯洰鏍囨帺鐩栨暟鐩
            break
        if index in covered_indexes:#閬垮厤閲嶅鎺╃洊浣嶇疆
            continue
        covered_indexes.add(index)

        masked_token = None

        #MASK绛栫暐 80%鏇挎崲涓篬MASK] 10%淇濇寔鑷韩涓嶅彉 10%鏇挎崲涓轰竴涓殢鏈虹殑token
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        # overwrite the above assignment with mask_prob
        if tokens[index] in MASK_PROB:#纭畾寰匨ASK鐨則oken鍦∕ASK_PROB涓笖浠1-MASK_PROB[tokens[index]]鐨勬鐜囪繘琛屾浛鎹
            if rng.random() < MASK_PROB[tokens[index]]:
                masked_token = tokens[index]
                # print(f'cover {tokens[index]} in correct instance.')
                #鏈鏇挎崲鏃剁粺璁+1
                CORRECT_COUNT[tokens[index]] += 1

        #杩涜鏇挎崲鎿嶄綔
        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    #缁熻涓涓猻equenc涓imask鎺夌殑浣嶇疆鍜屽搴旂殑label
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels

#瀵逛簬閿欒鍙ュ瓙鐨凪ASK绛栫暐
def create_masked_lm_predictions_for_wrong_sentences(tokens, masked_lm_prob,
                                                     max_predictions_per_seq, vocab_words, rng, wrong_tokens):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    #瀵硅緭鍏ョ殑token鍜寃rong_token鐨勯暱搴﹁繘琛屽垽鏂
    if not len(tokens) == len(wrong_tokens):
        print(tokens)
        print(wrong_tokens)
    assert len(tokens) == len(wrong_tokens)
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            assert wrong_tokens[i] == token
            continue
        elif token != wrong_tokens[i]:#灏嗛敊璇綅缃坊鍔犲埌寰呮浛鎹㈢殑閫夐」涓
            cand_indexes.append(i)
        else:  # when a token is not confused, add it to candidates according to its mask probability
            if token in MASK_PROB:
                if rng.random() < MASK_PROB[token]:
                    #鏇挎崲缁熻+1
                    WRONG_COUNT[token] += 1
                    # print(f'cover {token} in wrong instance.')
                    cand_indexes.append(i)#灏嗕綅缃坊鍔犲埌寰呮浛鎹㈢殑閫夐」涓

    rng.shuffle(cand_indexes)#index鎵撲贡

    output_tokens = list(tokens)

    # num_to_predict = min(max_predictions_per_seq,
    #                      max(1, int(round(len(tokens) * masked_lm_prob ))))
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens)))))  # we set 100% masking rate to allow all errors and corresponding non-errors to be masked

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        #鐢ㄨ嚜韬繘琛孧ASK
        masked_token = wrong_tokens[index]
        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels

#瀵瑰瓧娈佃繘琛屾埅鏂 浠ユ弧瓒砿ax_num_tokens
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng, wrong_tokens_a, wrong_tokens_b):
    """Truncates a pair of sequences to a maximum sequence length."""
    assert len(tokens_a) == len(wrong_tokens_a)
    try:
        assert len(tokens_b) == len(wrong_tokens_b)
    except:
        print(tokens_b)
        print(wrong_tokens_b)
        exit()

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        wrong_trunc_tokens = wrong_tokens_a if len(wrong_tokens_a) > len(wrong_tokens_b) else wrong_tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
            del wrong_trunc_tokens[0]
        else:
            trunc_tokens.pop()
            wrong_trunc_tokens.pop()


def main(_):
    #璁剧疆log鐨勫彲瑙佸害
    tf.logging.set_verbosity(tf.logging.INFO)
    #瀹炰緥鍖栧垎璇嶅櫒tokenizer 閲囩敤bert棰勮缁冪殑璇嶈〃
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file)
    #杈撳叆鏂囦欢锛坈orrect锛
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    #杈撳叆鏂囦欢锛坵rong锛
    wrong_input_files = []
    for wrong_input_pattern in FLAGS.wrong_input_file.split(","):
        wrong_input_files.extend(tf.gfile.Glob(wrong_input_pattern))

    print(input_files)

    tf.logging.info("*** Reading from input files ***")
    '''
    #鎴戣涓篶orrect鍜寃rong閮藉睘浜庤緭鍏ユ枃浠 鎵浠ヤ慨鏀逛簡杩欓儴鍒嗕唬鐮
    old version:
        for input_file in input_files:
            tf.logging.info("  %s", input_file)
    '''
    for input_file, wrong_input_file in zip(input_files, wrong_input_files):
        tf.logging.info("  %s", input_file)
        tf.logging.info("  %s", wrong_input_file)#鎴戣涓篶orrect鍜寃rong閮藉睘浜庤緭鍏ユ枃浠 鎵浠ヤ慨鏀逛簡杩欓儴鍒嗕唬鐮
    #鍥哄畾闅忔満鐢熸垚鐨勭瀛
    rng = random.Random(FLAGS.random_seed)
    #鏍规嵁杈撳嚭鍏ヨ緭鍑哄垱寤鸿缁冨疄渚
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, wrong_input_files)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("wrong_input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()




