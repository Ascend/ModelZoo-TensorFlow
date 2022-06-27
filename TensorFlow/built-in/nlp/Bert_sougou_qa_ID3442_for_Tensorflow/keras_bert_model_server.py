# -*- coding: utf-8 -*-
# @Time : 2021/1/7 11:21
# @Author : Jclian91
# @File : keras_bert_model_server.py
# @Place : Yangpu, Shanghai
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import traceback
from keras.backend import set_session
import numpy as np
from tokenizers import BertWordPieceTokenizer
from keras.models import load_model
from keras_bert import get_custom_objects
from flask import Flask
from flask import request, jsonify


max_len = 300

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("chinese_L-12_H-768_A-12/vocab.txt", lowercase=True)


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx
        answer = answer_text

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = [0]*max_len
        self.start_token_idx[start_token_idx] = 1
        self.end_token_idx = [0]*max_len
        self.end_token_idx[end_token_idx] = 1
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(question, context):
    squad_examples = []
    squad_eg = SquadExample(question, context, 0, "haha", "haha")
    squad_eg.preprocess()
    squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if not item.skip:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_ids"], dataset_dict["token_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.route("/model/bert_qa", methods=["GET", "POST"])
def get_geo():
    return_result = {"code": 200, "message": "success", "data": {}}
    try:
        input_context = request.get_json()["context"]
        input_question = request.get_json()["question"]

        test_squad_examples = create_squad_examples(input_question, input_context)
        x_train, y_train = create_inputs_targets(test_squad_examples)

        # 模型预测
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            pred_start, pred_end = model.predict(x_train)
        eval_examples_no_skip = [_ for _ in test_squad_examples if not _.skip]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            confidence = (pred_start[0][start]+pred_end[0][end])/2

            return_result["data"] = {"context": input_context, "question": input_question, "answer": pred_ans,
                                     "confidence": confidence}

    except Exception:
        return_result["code"] = 400
        return_result["message"] = traceback.format_exc()

    return jsonify(return_result)


if __name__ == '__main__':
    sess = tf.Session()
    graph = tf.get_default_graph()

    set_session(sess)
    # global model, sess, graph
    # 加载训练好的模型
    custom_objects = get_custom_objects()
    model = load_model("bert_sougou_qa.h5", custom_objects=custom_objects)
    # graph = tf.get_default_graph()
    # sess = tf.Session(graph=tf.Graph())
    # with sess.graph.as_default():
    #     keras.backend.set_session(sess)
    app.run(host="0.0.0.0", port=16500)

