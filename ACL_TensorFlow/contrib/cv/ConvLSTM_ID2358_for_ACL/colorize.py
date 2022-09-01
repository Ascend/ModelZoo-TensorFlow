# encoding: utf-8
import cv2
import numpy as np
import os
import sys
import time
import math
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
oppath = os.path.dirname(os.path.abspath(__file__))
print(oppath)
sys.path.append(os.path.join(oppath, "./acllite"))
#sys.path.append(os.path.join(oppath, "../../../../common/"))
#sys.path.append(os.path.join(oppath, "../../../../common/acllite"))

from numpy import mat
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource



model_path = './Convlstm_OM86.om'

def default_image_loader(path):
    img = cv2.imread(path)
    if img is not None:
        # Normalizing and Subtracting mean intensity value of the corresponding image
        img = img / np.max(img)
        img = img - np.mean(img)
        img = cv2.resize(img, (1280, 384), fx=0, fy=0)
    return img
    
class VisualOdometryDataLoader(object):#preprocess
    def __init__(self, args, datapath, trajectory_length, loader=default_image_loader):
        self.args = args   
        self._current_initial_frame = 0
        self._current_trajectory_index = 0
        self.current_epoch = 0

        self.sequences = [4]

        self.base_path = datapath
        self.poses = self.load_poses()
        self.trajectory_length = len(self.sequences)
        self.loader = loader

    def get_image(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', '%02d' % sequence, 'image_0', '%06d' % index + '.png')
        image = self.loader(image_path)
        return image

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', ('%02d' % sequence) + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)
        return all_poses

    def _set_next_trajectory(self):
        if (self._current_trajectory_index < self.trajectory_length-1):
            self._current_trajectory_index += 1
        else:
            self.current_epoch += 1
            self._current_trajectory_index = 0

        self._current_initial_frame = 0

    def get_next_batch(self):
        img_batch = []
        label_batch = []

        poses = self.poses[self._current_trajectory_index]

        for j in range(self.args.bsize):
            img_stacked_series = []
            labels_series = []

            read_img = self.get_image(self.sequences[self._current_trajectory_index],
                                      self._current_initial_frame + self.args.time_steps)
            if (read_img is None): self._set_next_trajectory()

            for i in range(self._current_initial_frame, self._current_initial_frame + self.args.time_steps):
                img1 = self.get_image(self.sequences[self._current_trajectory_index], i)
                img2 = self.get_image(self.sequences[self._current_trajectory_index], i + 1)
                img_aug = np.concatenate([img1, img2], -1) #是numpy中对array进行拼接的函数
                img_stacked_series.append(img_aug)
                pose = self.get6DoFPose(poses[i + 1, :]) - self.get6DoFPose(poses[i, :])
                labels_series.append(pose)
            img_batch.append(img_stacked_series)
            label_batch.append(labels_series)
           
            self._current_initial_frame += self.args.time_steps
        label_batch = np.array(label_batch)
      
        img_batch = np.array(img_batch)
     
        return img_batch, label_batch

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def get6DoFPose(self, p):
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((pos, angles))

def postprocess(result_list):
#    print("len(every result_list) are :",len(result_list))
    for i in range(1):
       for j in range(32):
            fh = open("/home/test_user07/Convlstm/txtcsv/estimated.txt","a")#Storage file of 6d pose estimation value
            fh.write("%f %f %f %f %f %f\n"%(result_list[i][j,0],
                                            result_list[i][j,1],
                                            result_list[i][j,2],
                                            result_list[i][j,3],
                                            result_list[i][j,4],
                                            result_list[i][j,5]))
            fh.close()

def cal_absolute_from_relative(xyz_euler):#6D-12D  
    
    xyz_euler = np.array(xyz_euler)
    pose_absolute = []
    t1 = mat(np.eye(4))
    print("//////////////////////////////////")
    for i in range(len(xyz_euler)):  
        x12 = xyz_euler[i, 0]
        y12 = xyz_euler[i, 1]
        z12 = xyz_euler[i, 2]
        theta1 = xyz_euler[i, 3] / 180 * np.pi
        theta2 = xyz_euler[i, 4] / 180 * np.pi
        theta3 = xyz_euler[i, 5] / 180 * np.pi
        tx = mat([[1, 0, 0], [0, math.cos(theta1), -math.sin(theta1)], [0, math.sin(theta1), math.cos(theta1)]])
        ty = mat([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        tz = mat([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        tr = tz * ty * tx
        t12 = np.row_stack((np.column_stack((tr, [[x12], [y12], [z12]])), [0, 0, 0, 1])) #np.concatenate()和np.row_stack()、np.vstack()的效果一样。
        t2 = t1 * t12
        pose_absolute.append(np.array(t2[0: 3, :]).reshape(-1))  
        t1 = t2

    return pose_absolute

class MyArgs():
    def __init__(self, datapath, outputpath, bsize, trajectory_length, time_steps=1):
        self.datapath = datapath
        self.outputpath = outputpath
        self.bsize = bsize
        self.trajectory_length = trajectory_length
        self.time_steps = time_steps

def main():
    """
    acl resource initialization
    """
    #if not os.path.exists(OUTPUT_DIR):
        #os.mkdir(OUTPUT_DIR)
    #ACL resource initialization    
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    model = AclLiteModel(model_path)
    '''
    images_list = [os.path.join(INPUT_DIR, img)
                   for img in os.listdir(INPUT_DIR)
                   if os.path.splitext(img)[1] in IMG_EXT]
    '''
    args = MyArgs(datapath='/home/test_user07/Convlstm/dataset/',#data set
               outputpath = '/home/test_user07/Convlstm/outputs/',
               bsize=32,
               trajectory_length=1,
               time_steps=1)
    data_loader = VisualOdometryDataLoader(args, args.datapath, args.trajectory_length)
    datas = []
    while data_loader.current_epoch < 1:
        batch_x, batch_y = data_loader.get_next_batch()
        datas.append(batch_x)
        print('Current epoch : %d' % data_loader.current_epoch)
        print('output length : %d' % len(datas))
    
    images_list = datas
    print("len images_list : ",len(images_list))
    for pic in images_list:#Calculate 8 lists
        print(np.array(pic).shape,pic.size)
        pic = np.float32(pic)
        print(pic.dtype)
        print(pic.shape,pic.size)
        print("len(input_list)  ",len(pic))
        result_list = model.execute([pic,])
        postprocess(result_list)

    #Organize the data
    file_oldd = open("/home/test_user07/Convlstm/txtcsv/estimated.txt", 'rb+')
    liness = file_oldd.readlines()
    file_oldd.seek(-len(liness[-1]), os.SEEK_END)
    file_oldd.truncate()  
    file_oldd.close()
    #6Dto12D_pose
    eulerr = open('/home/test_user07/Convlstm/txtcsv/estimated.txt')
    liness = eulerr.readlines()
    line_inn = []
    for line in liness:
            line = line.strip("\n")
            line = line.split(" ") 
            line = [float(x) for x in line]
            line_inn.append(line)
    pose_absolutee = cal_absolute_from_relative(line_inn)
    with open('/home/test_user07/Convlstm/txtcsv/estimated_12D_file.txt', 'w') as f:
             for i  in range(len(pose_absolutee)):
                 x1 =  pose_absolutee[i][0]
                 x2 =  pose_absolutee[i][1]
                 x3 =  pose_absolutee[i][2]
                 x  =  pose_absolutee[i][3]
                 y1 =  pose_absolutee[i][4]
                 y2 =  pose_absolutee[i][5]
                 y3 =  pose_absolutee[i][6]
                 y  =  pose_absolutee[i][7]
                 z1 =  pose_absolutee[i][8]
                 z2 =  pose_absolutee[i][9]
                 z3 =  pose_absolutee[i][10]
                 z  =  pose_absolutee[i][11]
                 f.write("%f %f %f %f %f %f %f %f %f %f %f %f\n" %(x1,x2,x3,x,y1,y2,y3,y,z1,z2,z3,z))

    print("Execute end")

if __name__ == '__main__':
    main()
