#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy import *
from numpy import mat
import math

"""根据6D相对位姿计算12D绝对位姿"""
def cal_absolute_from_relative(xyz_euler):  
    
    xyz_euler = np.array(xyz_euler)#利用numpy.array生成数组会更高效，numpy更适合数值运算。
    pose_absolute = []  # 12-d
    t1 = mat(np.eye(4))
    # pose_absolute.extend([np.array(t1[0: 3, :]).reshape([-1])])#对前三行进行一行的拼接，此处不需要
    # print(xyz_euler)
    print("//////////////////////////////////")
    # xyz_euler[1] = xyz_euler[1].split(" ")
    # print(xyz_euler[1].split()[2])
    # print(len(xyz_euler[1]))
    # print(xyz_euler[1])

    # xyz_euler.reshape(len(xyz_euler), 6)
    for i in range(len(xyz_euler)):  
        x12 = xyz_euler[i, 0]
        y12 = xyz_euler[i, 1]
        z12 = xyz_euler[i, 2]
        theta1 = xyz_euler[i, 3] / 180 * np.pi#角度转弧度
        theta2 = xyz_euler[i, 4] / 180 * np.pi
        theta3 = xyz_euler[i, 5] / 180 * np.pi
        tx = mat([[1, 0, 0], [0, math.cos(theta1), -math.sin(theta1)], [0, math.sin(theta1), math.cos(theta1)]])#生成矩阵
        ty = mat([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        tz = mat([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        tr = tz * ty * tx#进行单轴旋转矩阵相乘，得到旋转矩阵
        t12 = np.row_stack((np.column_stack((tr, [[x12], [y12], [z12]])), [0, 0, 0, 1]))#矩阵拼接,先列后行
        t2 = t1 * t12#相对的量相乘---->绝对的量？
        #print("t2 is :  ",t2)
        #t2 = np.delete(t2, 3, 0)
        #pose_absolute.append(np.array(t2).reshape(-1))
        #print("pose_ab is :", pose_absolute)
        pose_absolute.append(np.array(t2[0: 3, :]).reshape(-1))#对前三行进行一行的拼接
        
        t1 = t2

    return pose_absolute

if __name__ == '__main__':
        euler = open('/home/TestUser06/convlstm/npu1_4/txtcsv/output_file.txt')#真值文件转换
        lines = euler.readlines()
        line_in = []
        for line in lines:
            line = line.strip("\n")#删除换行
            line = line.split(" ") #按空格切片
            line = [float(x) for x in line]#将字符串转换为数值
            line_in.append(line)
        pose_absolute = cal_absolute_from_relative(line_in)    
        with open('/home/TestUser06/convlstm/npu1_4/txtcsv/output_12D_file.txt', 'w') as f:
             for i  in range(len(pose_absolute)):
                 x1 =  pose_absolute[i][0]
                 x2 =  pose_absolute[i][1]
                 x3 =  pose_absolute[i][2]
                 x  =  pose_absolute[i][3]
                 y1 =  pose_absolute[i][4]
                 y2 =  pose_absolute[i][5]
                 y3 =  pose_absolute[i][6]
                 y  =  pose_absolute[i][7]
                 z1 =  pose_absolute[i][8]
                 z2 =  pose_absolute[i][9]
                 z3 =  pose_absolute[i][10]
                 z  =  pose_absolute[i][11]
                 f.write("%f %f %f %f %f %f %f %f %f %f %f %f\n" %(x1,x2,x3,x,y1,y2,y3,y,z1,z2,z3,z))
        eulerr = open('/home/TestUser06/convlstm/npu1_4/txtcsv/estimated.txt')
        liness = eulerr.readlines()
        line_inn = []
        for line in liness:
            line = line.strip("\n")#删除换行
            line = line.split(" ") #按空格切片
            line = [float(x) for x in line]#将字符串转换为数值
            line_inn.append(line)
        pose_absolutee = cal_absolute_from_relative(line_inn)
        with open('/home/TestUser06/convlstm/npu1_4/txtcsv/estimated_12D_file.txt', 'w') as f:#预测文件转换
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
        
