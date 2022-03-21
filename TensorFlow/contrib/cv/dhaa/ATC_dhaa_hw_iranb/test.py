import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import pprint
from pathlib import Path
import random
import sys

data_path = "/root/code/dhaa/LAJ_DHAA/ATC_DHAA/feat_bin/test/20210709_174724"
label_file = "/root/code/dhaa/LAJ_DHAA/morph/txt/pts_txt/morph_pts_80-20_test.txt"


def instance_evaluation(pre_labels_array, labels_array):
    MAE_total = np.mean(np.abs(pre_labels_array - labels_array))
    return MAE_total


def get_bin_label_dict(path, label_file):
    dataset = defaultdict(list)
    with open(label_file) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            img_name = contents[0]
            final_name_ = '_'.join(img_name.split('/'))
            final_name_ = os.path.splitext(final_name_)[0]
            final_name = "Cropped_91_145_"+final_name_ + '_output_0.bin'
            image_bin_path = os.path.join(path, final_name)
            labels = [int(contents[1])]

            # labels = np.array( labels, dtype=np.int )
            dataset[image_bin_path].append(labels)
    return dataset



out_dir = "/root/code/dhaa/LAJ_DHAA/ATC_DHAA/feat_bin/test/"

label_dim = 101
temp = np.reshape(np.arange(label_dim, dtype=np.float32), (label_dim, 1))
# for file in Path(out_dir).rglob('*.bin'):
#     this_out = np.fromfile(file, dtype=np.float32)
#     out = this_out.reshape(1,101)
#     result = np.matmul(out, temp)
#     print(result)
    

def main(test_bin_path, list_file, fid):
    #### Set GPU options ###
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    # config.gpu_options.allow_growth = True

    ### Setting Parameters ###
    # mean_div = []
    # mean_div += (float(i) for i in args.mean_div_str.split('-'))
    # label dim has been tranformed in the PB model
    label_dim = 101

    ### Make folders of logs and models ###
    # subdir = args.subdir ## datetime.strftime(datetime.now(), args.subdir)
    # scores_dir = os.path.join(args.train_url, 'scores', subdir)
    ### Load data ###
    img_label_dict = get_bin_label_dict(test_bin_path, list_file)
    # test_image_list, test_label_list = utils.get_image_paths_and_labels(data_test)
    print('Test: num_images={}/num_ID={}'.format(len(img_label_dict), len(img_label_dict)))
    
    # use om bin feature
    batch_size = 10
    AllN_s = len(img_label_dict)
    img_bin_path = list(img_label_dict.keys())
    
    random.shuffle(img_bin_path)
    nrof_batches = AllN_s // batch_size
    nrof_images = nrof_batches * batch_size
    test_batches = nrof_batches
    
    pre_labels_array = np.zeros((nrof_images, 1), np.float32)
    labels_array = np.zeros((nrof_images, 1), np.float32)
    ### Testing loop
    start_idx = 0
    for batch_it in range(test_batches):
        final_index = (batch_it+1)*batch_size if (batch_it+1)*batch_size < len(img_bin_path) else len(img_bin_path)
        start_index = batch_it*batch_size
        tmp_img_list = img_bin_path[start_index:final_index]
        img_feat_list, img_label_list = [], []
        for i in tmp_img_list:
            tmp_feature = np.matmul(np.fromfile(i, dtype='float32').reshape([1, 101]), temp)
            img_feat_list.append(tmp_feature)
            img_label_list.append(img_label_dict[i])
        img_feats = np.array(img_feat_list).reshape([-1, 1])
        
        img_labels = np.array(img_label_list).reshape([-1, 1])
        pre_labels_ = img_feats
        pre_labels_ = pre_labels_ > 0
        c_n = pre_labels_.shape[0]
        pre_labels_array[start_idx:start_idx + c_n] = pre_labels_
        labels_array[start_idx:start_idx + c_n] = img_labels
        start_idx += c_n
        if batch_it % 10 == 0:
            print('.', end='')
            sys.stdout.flush()
    print('')
    Result = instance_evaluation(pre_labels_array, labels_array)
    
    print(20*"=")
    print("MAE RESULT", Result)
    print(20*"=")
    
    # MeanAcc = 100*Result['accuracy_mean']
    # InsACC = 100*Result['instance_accuracy']
    # InsRecall = 100*Result['instance_recall']
    # InsPrec = 100*Result['instance_precision']
    # InsF1 = 100*Result['instance_F1']
    # fid.write('----------results--------\n')
    # fid.write('accuracy_mean: %.2f\tinstance_accuracy: %.2f\tinstance_recall: %.2f\tinstance_precision: %.2f\tinstance_F1: %.2f\t'
    #             %(MeanAcc, InsACC, InsRecall, InsPrec, InsF1))
    # print('accuracy_mean: %.2f\tinstance_accuracy: %.2f\tinstance_recall: %.2f\tinstance_precision: %.2f\tinstance_F1: %.2f\t'
    #             %(MeanAcc, InsACC, InsRecall, InsPrec, InsF1))
    # print('Test Ending!')

if __name__ == '__main__':
    fid = open('./Test_scores.txt', 'w')
    test_dataset_path = "/root/code/dhaa/LAJ_DHAA/morph/txt/pts_txt/morph_pts_80-20_test.txt"
    test_bin_path = "/root/code/dhaa/LAJ_DHAA/ATC_DHAA/feat_bin/test/20210709_174724"
    main(test_bin_path, test_dataset_path, fid)
    fid.close()
