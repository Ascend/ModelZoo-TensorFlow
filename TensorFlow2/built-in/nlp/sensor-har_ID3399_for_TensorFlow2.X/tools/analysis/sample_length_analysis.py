import os
import os.path as osp
import json
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from rich.console import Console

DATASET = 'zim'
CONSOLE = Console()


def clean(str):
    return str.replace('_', ' ').lower().strip()


def parse_args():
    parser = ArgumentParser(prog='analysis of sample length distribution')
    parser.add_argument('in_dir', help='directory with raw json files')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-10.txt',
        help='annotation file')
    parser.add_argument(
        '--dataset',
        type=str,
        default='zim',
        choices=['zim'],
        help='dataset')
    args = parser.parse_args()
    return args


def save_result(out, result, label):
    if result == {}:
        return
    df = pd.DataFrame({'Length': [k for k in result.keys()],
        'Value': [v for v in result.values()]})

    # save json
    keys = [int(k) for k in result.keys()]
    result['min'] = min([k for k in keys if k > 30])
    result['max'] = max(keys)
    result_json = json.dumps(result, indent=4)
    f = open(osp.join(out, 'sample_len_dist.json'), 'w')
    print(result_json, file=f)
    f.close()

    # save plot
    sns.set(rc={'figure.figsize': (15, 13)})
    fig = sns.barplot(x='Length', y='Value', data=df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=15)
    fig.axes.set_title(label, fontsize=40)
    fig.set_xlabel('Len', fontsize=30)
    fig.set_ylabel('Val', fontsize=20)
    output = fig.get_figure()
    output.savefig(osp.join(out, 'sample_len_dist.svg'))


def main():
    args = parse_args()
    out_dir = osp.join('results/sample_len/', args.dataset)
    labels = []
    with open(args.ann, 'r') as ann:
        for line in ann:
            _, label = line.split(' ', 1)
            labels.append(label.strip())
    for label in labels:
        Path(osp.join(out_dir, label)).mkdir(parents=True, exist_ok=True)

    for label in labels:
        result = {}
        # CONSOLE.print(label, style='green')
        for file in os.listdir(args.in_dir):
            file_label = clean(file.split('2021')[0])
            if file_label != label:
                continue
            # CONSOLE.print(file_label, style='yellow')

            content = open(osp.join(args.in_dir, file), 'r')
            length = len(json.load(content))

            # if file_label == 'english valse natural turn bw 1 - 3':
            #     if length == 103:
            #         CONSOLE.print(file, style='green')

            if result.get(length, None) is None:
                result[length] = 1
            else:
                result[length] += 1

        save_result(osp.join(out_dir, label), result, label)
    CONSOLE.print('Saved plots', style='green')


if __name__ == '__main__':
    main()
