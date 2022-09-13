import csv
import yaml
import os.path as osp

from argparse import ArgumentParser
from rich.console import Console

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(
        prog='get the size of train/val/test splits')
    parser.add_argument(
        '--dataset',
        type=str,
        default='zim',
        choices=['zim', 'pamap2'],
        help='dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file = open('configs/data.yaml', 'r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config[args.dataset]
    if args.dataset == 'zim':
        path = osp.join(data_config['data_dir']['raw'], args.dataset, 'merged')
    elif args.dataset == 'pamap2':
        path = osp.join(data_config['data_dir']['raw'], args.dataset,
                        'PAMAP2_Dataset', 'Protocol')

    result = {'train_files': 0, 'validation_files': 0, 'test_files': 0}
    for key in result.keys():
        # CONSOLE.print(key, style='green')
        for dat in config[key]:
            c_path = osp.join(path, dat)

            with open(c_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                for _ in csv_reader:
                    pass
                result[key] += csv_reader.line_num

    tot = sum(result.values())
    for k, v in result.items():
        CONSOLE.print(f'{k.split("_files")[0]}: {100*round(v/tot, 2)}%', style='bold green')


if __name__ == '__main__':
    main()
