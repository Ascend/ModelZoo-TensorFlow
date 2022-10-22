import numpy as np
import glob
import pathlib
import os
import utils_gpu
import tensorflow as tf

INFERENCE_DIR = "C:/Users/1young/Desktop/svd_output/svd_output/*.bin"
TEST_DIR = "D:/svd_code2/data_1"
tf.enable_eager_execution()


def main():
    input_test_files = glob.glob(INFERENCE_DIR)
    mean_err = []
    for in_file in input_test_files:
        out_file_prefix = pathlib.Path(in_file).stem
        rot_path = os.path.join(TEST_DIR,'%s.txt' % out_file_prefix)
        rot = np.loadtxt(rot_path)[:3,:].reshape((-1,3,3))
        r = np.loadtxt(in_file).reshape((-1,3,3))
        theta = utils_gpu.relative_angle(rot, r)
        mean_theta = tf.reduce_mean(theta)
        mean_theta_deg = mean_theta * 180.0 / np.pi
        mean_theta_deg = mean_theta_deg.numpy()
        mean_err.append(mean_theta_deg)
    print("the mean of error")
    print(np.mean(np.array(mean_err)))

if __name__=="__main__":
    main()



