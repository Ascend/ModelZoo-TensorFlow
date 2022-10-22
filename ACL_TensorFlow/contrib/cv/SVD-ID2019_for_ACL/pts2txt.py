import os
import numpy as np
import tensorflow as tf
import glob
import pathlib

inputpath = "D:\special_orthogonalization\points_test_modified\*.pts"
outputpath = "D:\svd_code2\data_1"


tf.enable_eager_execution()

def data_processing(pts_path):
    file_buffer = tf.read_file(pts_path)
    lines = tf.string_split([file_buffer], delimiter='\n')
    lines1 = tf.string_split(lines.values, delimiter='\r')
    values = tf.stack(tf.decode_csv(lines1.values,
                                    record_defaults=[[0.0], [0.0], [0.0]], field_delim=' '))
    values = tf.transpose(values)  # 3xN --> Nx3.
    diff_num = 1414-tf.shape(values)[0]
    repeat_pts = tf.tile(tf.reshape(values[4,:],(1,-1)),[diff_num,1])

    values = tf.concat([values,repeat_pts],axis=0)
    # First three rows are the rotation matrix, remaining rows the point cloud.
    values = tf.concat([values, repeat_pts], axis=0)
    # First three rows are the rotation matrix, remaining rows the point cloud.
    return values.numpy()

def file_save(path,datapath):
    input_test_files = glob.glob(path)
    for in_file in input_test_files:
        out_file_prefix = pathlib.Path(in_file).stem
        values = data_processing(in_file)
        out_file1 = os.path.join(
            datapath, '%s.txt' % out_file_prefix)
        np.savetxt(out_file1,values)

def main():
    os.makedirs(outputpath, exist_ok=True)
    file_save(inputpath,outputpath)

if __name__ == '__main__':
    main()

