import csv
import json
import yaml
import os
import os.path as osp

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from rich.console import Console

LABEL_TO_NUMBER = {}
CONSOLE = Console()
TOLERANCE = 3


def clean(str):
    return str.replace('_', ' ').lower()


def parse_args():
    parser = ArgumentParser(
        prog='merge single samples (all classes) of one into 1 data file')
    parser.add_argument('in_dir', type=str, help='directory of samples')
    parser.add_argument(
        '--out',
        type=str,
        default='data/raw/zim/merged/subjectXY.dat',
        help='resulting file')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-valse.txt',
        help='annotation file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    global LABEL_TO_NUMBER
    with open(args.ann, 'r') as ann:
        for line in ann:
            (val, key) = line.split(' ', 1)
            LABEL_TO_NUMBER[key.strip()] = int(val)
    data_config = yaml.load(
        open('configs/data_proc.yaml', 'r'),
        Loader=yaml.FullLoader)['zim']

    skip_count = 0
    with open(args.out, mode='w') as out:
        writer = csv.writer(out, delimiter=' ')

        for file in tqdm(os.listdir(args.in_dir)):
            label = clean(file.split('2021')[0])
            activity_id = LABEL_TO_NUMBER.get(label.strip(), None)
            if activity_id is None:
                continue
            content = open(osp.join(args.in_dir, file), 'r')
            content = json.load(content)

            lower_b, upper_b = data_config[f'class_{activity_id}']
            if (len(content)+TOLERANCE < lower_b) | (len(content)-TOLERANCE > upper_b):
                CONSOLE.print(f'Skipping {file} with len {len(content)}',
                    f'Out of Bounds {lower_b} - {upper_b}', style='yellow')
                skip_count += 1
                continue

            for row in content:
                result = [activity_id]
                result.extend([x for x in row])
                writer.writerow(result)

    CONSOLE.print(f'Finished. Skipped {skip_count} samples')


if __name__ == '__main__':
    main()
