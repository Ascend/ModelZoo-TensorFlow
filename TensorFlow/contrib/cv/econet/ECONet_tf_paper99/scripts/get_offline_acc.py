import numpy as np
import argparse
import pickle
import os


parser = argparse.ArgumentParser(description="Offline Inference Accuracy Computation")
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)


def main():
    args = parser.parse_args()

    label_path = args.label_path
    output_path = args.output_path

    with open(label_path, 'rb') as f:
        label_dict = pickle.load(f)

    output_num = 0.
    check_num = 0.
    print("Start accuracy computation")
    for par, dir_list, file_list in os.walk(output_path):
        for file_name in file_list:
            if file_name.endswith('_output_0.bin'):
                output_num += 1
                output_logit = np.fromfile(os.path.join(par, file_name), dtype='float32')
                inf_label = int(np.argmax(output_logit))
                img_key = file_name.replace('_output_0.bin', '')
                print("%s, inference label:%d, gt_label:%d"%(img_key, inf_label, label_dict[img_key]))
                if inf_label == label_dict[img_key]:
                    check_num += 1

    top1_acc = check_num / output_num
    print("Totol image num: %d, Top1 accuarcy: %.4f"%(output_num, top1_acc))


if __name__ == '__main__':
    main()
