import numpy as np
import os
import cv2
from tqdm import tqdm


name_list = os.listdir("./output_final")
path = "./cartoonized"

if not os.path.exists(path):
    os.makedirs(path)

for item in tqdm(name_list):
    name = os.path.join("./output_final",item)
    with open(name) as f:
        file = f.read().split()
    file = np.array(file)

    file = np.array(file).reshape(1,256,256,3).astype(np.float32)
    save_path = os.path.join(path, item.replace("_output_0.txt", ''))

    output = (np.squeeze(file) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, output)

