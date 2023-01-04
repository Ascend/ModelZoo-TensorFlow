import sys
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    bin_dir = Path(args.bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    assert img_dir.exists(), 'img_dir is not exists!!!'

    for file in tqdm(list(img_dir.glob('*.*'))):
        img = cv2.imread(str(file))
        img.tofile(bin_dir / (file.name.rsplit('.')[0] + '.bin'))

    print('img to bin solve over!!!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", dest="img_dir", default="../coco/images")
    parser.add_argument("--bin-dir", dest="bin_dir", default="../coco/input_bins")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
