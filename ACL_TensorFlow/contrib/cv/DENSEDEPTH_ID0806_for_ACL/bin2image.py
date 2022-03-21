import numpy as np
from acl_utils import DepthNorm, display_images
import argparse
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--input', default=r'./test.bin', type=str, help='input file path.')
parser.add_argument('--output', default=r'./test.png', type=str, help='Output file.')
parser.add_argument('--bs', default=1, type=int, help='Batch size.')
args = parser.parse_args()

maxDepth = 1000
minDepth = 10
image = np.fromfile(args.input, dtype=np.float32)
print(image.shape)
print(type(image))
print(len(image))
image = np.reshape(image, (args.bs, 240, 320, 1))
image = np.clip(DepthNorm(image, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
print(image.shape)

viz = display_images(image.copy())
plt.figure(figsize=(10, 5))
plt.imshow(viz)
plt.savefig(args.output)
plt.show()
