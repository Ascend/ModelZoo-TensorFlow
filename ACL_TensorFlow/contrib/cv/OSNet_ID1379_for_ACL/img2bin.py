import numpy as np
import random
from sklearn.utils import shuffle as shuffle_tuple
import os
import cv2

random.seed(2021)

def get_data_information(data_root):
    img_path_list = []
    img_name_list = []
    img_cams_list = []
    image_names = os.listdir(data_root)  # the best way is to use sorted list,i.e., sorted()
    image_names = sorted(image_names)[:-1]
    for item in image_names:
        if item[-4:] == '.jpg':
            img_path_list.append(os.path.join(data_root, item))
            img_name_list.append(item.split('_')[0])
            img_cams_list.append(item.split('c')[1][0])
    return img_path_list, np.array(img_name_list), np.array(img_cams_list)


def main():
    batch_size = 1
    img_width = 64
    img_height = 128
    query_img_list, query_name_list, query_cams_list = get_data_information('Market-1501-v15.09.15/query')
    gallery_img_list, gallery_name_list, gallery_cams_list = get_data_information('Market-1501-v15.09.15/bounding_box_test')

    img_path_list = query_img_list
    M = len(query_img_list)
    N = len(gallery_img_list)

    query_index = 0
    while query_index <= M-1:
        current_index = (query_index * batch_size) % N  # the first index for each batch per epoch
        current_batch_size = batch_size
        query_index += 1  # indicates the next batch_size
        img_batch_list = query_img_list[current_index:current_index + current_batch_size]

        query_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        for i, img_path in enumerate(img_batch_list):
            img = cv2.imread(img_path)
            if img.shape[:2] != (img_height, img_width):
                img = cv2.resize(img, (img_width, img_height))
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            query_batch[i, :, :, :] = img
        # normalization
        query_batch = query_batch / 255.
        query_batch = (query_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        query_batch.tofile('query/query_{index}.bin'.format(index=query_index-1))


    gallery_index = 0
    while gallery_index <= N - 1:
        current_index = (gallery_index * batch_size) % N  # the first index for each batch per epoch
        current_batch_size = batch_size
        gallery_index += 1  # indicates the next batch_size
        img_batch_list = gallery_img_list[current_index:current_index + current_batch_size]

        gallery_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        for i, img_path in enumerate(img_batch_list):
            img = cv2.imread(img_path)
            if img.shape[:2] != (img_height, img_width):
                img = cv2.resize(img, (img_width, img_height))
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            gallery_batch[i, :, :, :] = img
        # normalization
        gallery_batch = gallery_batch / 255.
        gallery_batch = (gallery_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        gallery_batch.tofile('gallery/gallery_{index}.bin'.format(index=gallery_index - 1))

    print("Done")
if __name__ == "__main__":
    main()