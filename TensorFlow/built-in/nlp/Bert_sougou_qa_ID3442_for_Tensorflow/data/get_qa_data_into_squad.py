# -*- coding: utf-8 -*-
# @Time : 2021/1/5 14:45
# @Author : Jclian91
# @File : get_qa_data_into_squad.py
# @Place : Yangpu, Shanghai
import json
import uuid
import random
from random import shuffle

with open("SogouQA.json", "r", encoding="utf-8") as f:
    content = json.loads(f.read())

random.seed(2021)
shuffle(content)

squad_train_data = {"data": []}
for line in content[:int(len(content)*0.9)]:
    question = line["question"]
    question_id = line["id"]
    for para in line["passages"]:
        context = para["passage"]
        title = context[:20]
        answer = para["answer"]
        if answer:
            answer_start = context.index(answer)
            # print(context, answer, answer_start)

            squad_train_data["data"].append({"title": title,
                                       "paragraphs": [
                                           {
                                            "context": context,
                                            "qas": [{
                                                "answers": [{"answer_start": answer_start, "text": answer}],
                                                "question": question,
                                                "id": question_id
                                            }]
                                            }
                                       ]
                                       })


with open("sogou_qa_train.json", "w", encoding="utf-8") as g:
    g.write(json.dumps(squad_train_data, ensure_ascii=False, indent=4))

squad_test_data = {"data": []}
for line in content[int(len(content)*0.9):]:
    question = line["question"]
    question_id = line["id"]
    for para in line["passages"]:
        context = para["passage"]
        title = context[:20]
        answer = para["answer"]
        if answer:
            answer_start = context.index(answer)
            # print(context, answer, answer_start)

            squad_test_data["data"].append({"title": title,
                                       "paragraphs": [
                                           {
                                            "context": context,
                                            "qas": [{
                                                "answers": [{"answer_start": answer_start, "text": answer}],
                                                "question": question,
                                                "id": question_id
                                            }]
                                            }
                                       ]
                                       })


with open("sogou_qa_test.json", "w", encoding="utf-8") as g:
    g.write(json.dumps(squad_test_data, ensure_ascii=False, indent=4))

print(len(squad_train_data["data"]))
print(len(squad_test_data["data"]))

