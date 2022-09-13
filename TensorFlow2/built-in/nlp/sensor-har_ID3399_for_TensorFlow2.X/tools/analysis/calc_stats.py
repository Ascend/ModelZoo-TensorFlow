import json
import numpy as np
from scipy import stats

from argparse import ArgumentParser
from rich.console import Console

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(
        prog='calculate sample statistics')
    parser.add_argument('sample', help='sample path')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='acc',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    parser.add_argument(
        '--mode',
        type=str,
        default='normal',
        choices=['normal', 'steps'],
        help='recording mode')
    parser.add_argument(
        '--steps',
        type=int,
        default=3,
        help='number of steps that the movement has')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
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
    if args.mode == 'steps':
        for k in sensors:
            results[k] = {}
            for i in range(1, args.steps+1):
                results[k][f'Step_{i}'] = []

        for row in content:
            step, data = row
            for i in range(len(sensors)):
                results[ind_to_sensor[i+balancer]][step].append(data[i+balancer])

        for s in sensors:
            for i in range(1, args.steps+1):
                step = f'Step_{i}'

                CONSOLE.print(
                        f'Sensor: {s} | Step: {step}| length: {len(results[s][step])}',
                        # f'Mean: {np.mean(results[s][step])} \n',
                        # f'groupped 25th percentile: {np.mean(np.percentile(results[s][step], [21, 22, 23, 24, 25, 26, 27]))}',
                        # f'25th percentile: {np.percentile(results[s][step], 25)} \n',
                        # f'Median: {np.median(results[s][step])}',
                        # f'75th percentile: {np.percentile(results[s][step], 75)} \n',
                        # f'Mode: {stats.mode([round(r, 2) for r in results[s][step]])}',
                        # f'Mode: {stats.mode([round(r, 2) for r in results[s][step]])[0][0]}',
                        # f'Std: {np.std(results[s][step])}',
                        f'Min: {np.min(results[s][step])}',
                        f'Max: {np.max(results[s][step])}', style='green')

    else:
        for row in content:
            for i in range(len(sensors)):
                results[ind_to_sensor[i+balancer]].append(row[i+balancer])

        for s in sensors:
            CONSOLE.print(
                        f'Sensor: {s} | Mean: {np.mean(results[s])} \n',
                        f'groupped 25th percentile: {np.mean(np.percentile(results[s], [21, 22, 23, 24, 25, 26, 27]))}',
                        f'25th percentile: {np.percentile(results[s], 25)} \n',
                        f'Median: {np.median(results[s])}',
                        f'75th percentile: {np.percentile(results[s], 75)} \n',
                        f'Mode: {stats.mode([round(r, 2) for r in results[s]])}',
                        f'Mode: {stats.mode([round(r, 2) for r in results[s]])[0][0]}',
                        f'Std: {np.std(results[s])}', style='green')


if __name__ == '__main__':
    main()
