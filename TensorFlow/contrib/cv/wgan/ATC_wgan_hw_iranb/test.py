import numpy as np
from pathlib import Path
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import math


from skimage.metrics import structural_similarity as ssim


def Average(lst):
    return sum(lst) / len(lst)


def data2img(data):
    rescaled = np.divide(data + 1.0, 2.0)
    return np.reshape(np.clip(rescaled, 0.0, 1.0), [1, 64, 64, 3])


i = 0
result_scroe = []
dataset_path = "/root/dataset/lsunbin/dataset_bin"
out_feature_path = "/root/code/wgan/ATC_wgan/feat_bin/20210725_161841"
# for result in Path(out_feature_path).rglob("*.bin"):
#     test_result = np.fromfile(result, dtype=np.float64)
#     test_result = np.reshape(test_result, (1, 64, 64, 3))
#     test_recon_image = data2img(test_result)
#     test_recon_image = np.reshape(test_recon_image, (64, 64, 3))
#     current = []
#     for data in dataset:
#         print(data)
#         print(ssim(test_recon_image * 255, data, multichannel=True))
#         exit(0)
#     # print(ssim(test_recon_image, test_recon_image, multichannel=True))


out = []
for data in Path(dataset_path).rglob("*.bin"):
    current_data = np.fromfile(data, dtype=np.float32)
    current_data = np.reshape(current_data, (64, 64, 3))
    current = []
    for feature in Path(out_feature_path).rglob("*.bin"):
        # for test image
        test_result = np.fromfile(feature, dtype=np.float32)
        test_result = np.reshape(test_result, (1, 64, 64, 3))
        test_recon_image = data2img(test_result)
        test_recon_image = np.reshape(test_recon_image, (64, 64, 3))
        # for data
        ss = ssim(test_recon_image, current_data, multichannel=True)
        current.append(math.fabs(ss))

    out.append(Average(current))

print("mean ss :", 15 * "=")
print(Average(out))
print(20 * "=")
