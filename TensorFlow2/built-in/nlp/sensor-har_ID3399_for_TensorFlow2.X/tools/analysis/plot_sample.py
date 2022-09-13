import json
import string

import pandas as pd
import seaborn as sns
import os.path as osp
import random as rd

from argparse import ArgumentParser
from rich.console import Console
from pathlib import Path


DATASET = 'zim'
CONSOLE = Console()


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def clean(str):
    return str.replace('_', ' ').lower()


def parse_args():
    parser = ArgumentParser(
        prog='plot single sample')
    parser.add_argument('sample', help='sample path')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='acc',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=f'results/samples/{DATASET}',
        help='out dir to save results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.sensor_type == 'both':
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
        balancer = 0
    elif args.sensor_type == 'acc':
        sensors = ['acc_x', 'acc_y', 'acc_z']
        balancer = 0
    else:
        sensors = ['gy_x', 'gy_y', 'gy_z']
        balancer = 3
    ind_to_sensor = {i+balancer: sensor for i, sensor in enumerate(sensors)}
    content = json.load(open(args.sample, 'r'))
    results = {k: [] for k in sensors}

    palette = ['Reds', 'Blues', 'Greys', 'Oranges', 'Purples', 'Greens']

    for row in content:
        for i in range(len(sensors)):
            results[ind_to_sensor[i+balancer]].append(row[i+balancer])

    sns.set(rc={'figure.figsize': (15, 13)})
    for i, k in enumerate(results.keys()):
        df = pd.DataFrame({'Sample Index': list(range(0, len(results[k]))),
                    'Sensor Reading': results[k], 'Sensor': k})
        fig = sns.lineplot(x='Sample Index', y='Sensor Reading', data=df, hue='Sensor', palette=palette[i])
        fig.set_xlabel('Sample Index', fontsize=30)
        fig.set_ylabel('Sensor Reading', fontsize=20)

    output = fig.get_figure()
    movement, person = args.sample.split('/')[-1], args.sample.split('/')[-2]
    movement = clean(movement).split(' ')
    movement = '-'.join(movement[i] for i in range(0, 5))
    output.savefig(osp.join(args.out_dir,
        f'{movement}_{args.sensor_type}_{person}_{gen_id(3)}.svg'), bbox_inches='tight')
    CONSOLE.print('Saved plot', style='green')


if __name__ == '__main__':
    main()
