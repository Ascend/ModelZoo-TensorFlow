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
import numpy as np
import os, argparse
import torch
from utils import dataset
from collections import defaultdict
from utils.evaluator import CZSL_Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to msame inference result")
    parser.add_argument("--data_url", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--data", type=str, default='UT', choices=['MIT', 'UT', 'MITg', 'UTg'], help="Dataset name")
    parser.add_argument("--test_bz", type=int, default=1, help="Test batch size")
    parser.add_argument("--obj_pred", type=str, default=None, help="Object prediction from pretrained model")
    return parser


def load_txt(path, type):
    with open(path, 'r')as f:
        line = f.readline().strip().split(" ")

    predict = np.asarray(line, dtype=type).reshape(1, -1)
    return predict


def formated_czsl_result(report):
    fstr = 'rA:{real_attr_acc:.4f}|rO:{real_obj_acc:.4f}|Cl/T1:{top1_acc:.4f}|T2:{top2_acc:.4f}|T3:{top3_acc:.4f}'
    return fstr.format(**report)


def main():
    parser = make_parser()
    args = parser.parse_args()
    test_dataloader = dataset.get_dataloader(args.data_url, args.data, 'test', batchsize=args.test_bz,
                                             obj_pred=args.obj_pred)
    dset = test_dataloader.dataset
    test_att_id = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
    test_obj_id = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])

    evaluator = CZSL_Evaluator(test_dataloader.dataset, None)

    accuracies_pair = defaultdict(list)
    accuracies_attr = defaultdict(list)
    accuracies_obj = defaultdict(list)

    for idx, batch in enumerate(test_dataloader):
        id = "{0:05d}_output_".format(idx)
        prob_P_rmd = load_txt(os.path.join(args.input, id + "0.txt"))
        prob_A_attr = load_txt(os.path.join(args.input, id + "1.txt"))
        prob_O = load_txt(os.path.join(args.input, id + "2.txt"))
        score = dict([
            ("score_rmd", [prob_P_rmd, prob_A_attr, prob_O]),  # Mul_18, Softmax_3, Placeholder_6
        ])

        for key in score.keys():
            score[key][0] = {
                (a, o): torch.from_numpy(score[key][0][:, j])
                for j, (a, o) in enumerate(zip(test_att_id, test_obj_id))
            }

        prediction = score
        attr_truth, obj_truth = batch[1], batch[2]
        attr_truth, obj_truth = torch.from_numpy(attr_truth), torch.from_numpy(obj_truth)

        for key in prediction.keys():
            p_pair, p_a, p_o = prediction[key]
            pair_results = evaluator.score_model(p_pair, obj_truth)
            match_stats = evaluator.evaluate_predictions(pair_results, attr_truth, obj_truth)
            accuracies_pair[key].append(match_stats)  # 0/1 sequence of t/f

            a_match, o_match = evaluator.evaluate_only_attr_obj(p_a, attr_truth, p_o, obj_truth)

            accuracies_attr[key].append(a_match)
            accuracies_obj[key].append(o_match)

    for name in accuracies_pair.keys():
        accuracies = accuracies_pair[name]
        accuracies = zip(*accuracies)
        accuracies = map(torch.mean, map(torch.cat, accuracies))
        attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, objoracle_acc = map(lambda x: x.item(),
                                                                                            accuracies)

        real_attr_acc = torch.mean(torch.cat(accuracies_attr[name])).item()
        real_obj_acc = torch.mean(torch.cat(accuracies_obj[name])).item()

        report_dict = {
            'real_attr_acc': real_attr_acc,
            'real_obj_acc': real_obj_acc,
            'top1_acc': closed_1_acc,
            'top2_acc': closed_2_acc,
            'top3_acc': closed_3_acc,
        }

        print(name + ": " + formated_czsl_result(report_dict))

    pass


if __name__ == '__main__':
    main()

