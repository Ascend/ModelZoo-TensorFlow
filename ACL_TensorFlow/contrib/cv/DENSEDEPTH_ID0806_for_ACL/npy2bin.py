import numpy as np
from acl_utils import extract_zip
from io import BytesIO
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--nyu_dir', default=r'./dataset/nyu_test.zip', type=str, help='input file path.')
parser.add_argument('--output', default=r'./result/bin', type=str, help='Output folder.')
parser.add_argument('--bs', default=4, type=int, help='Batch size.')
args = parser.parse_args()

print('Start changing npy images to bin...')
print('Loading test data...')

data = extract_zip(args.nyu_dir)
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
print("The dtype of images [{}]".format(rgb.dtype))
# depth = np.load(BytesIO(data['eigen_test_depth.npy']))
# crop = np.load(BytesIO(data['eigen_test_crop.npy']))

nums = len(rgb)
bs = args.bs

print('The numbers of images [{}]'.format(nums))

# pretreatment
rgb = rgb / 255
rgb = rgb.astype(np.float32)

print('After pretreatment, the dtype of images [float32]')

count = 0

output_path = os.path.join(args.output, "image")
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path_flip = os.path.join(args.output, "image_flip")
if not os.path.exists(output_path_flip):
    os.makedirs(output_path_flip)

for i in range(nums // bs):
    x = rgb[i * bs:(i + 1) * bs, :, :, :]
    output_file_path = os.path.join(output_path, "rgb_bs{}_{}.bin".format(args.bs, i))
    x.tofile(output_file_path)

    # flip images bin(mirror image)
    output_file_path_flip = os.path.join(output_path_flip, "rgb_flip_bs{}_{}.bin".format(args.bs, i))
    x = x[..., ::-1, :]
    x.tofile(output_file_path_flip)

    count = i

print('The numbers of bin [{}]'.format(count + 1))

print('Successful')
