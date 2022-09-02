"""
以此构建一个我自己的数据集处理过程
将处理后的数据用numpy保存
其他文件直接读取即可
给予其他程序一个可选接口 即用以保存文件 还是重新读取 保存 然后返回读取的数据集。
重新读取比较耗时。直接读取保存的比较快。

训练集和测试集一同读取 返回
(train_imgs,train_labels),(test_imgs,test_labels)
接受3个参数 1个必须 2个可选
1 是否从原文件重新读取保存
    2 是否归一化
    3 是否one_hot
he data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files.
All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.
数据以大端格式存放而intel处理器是小端模式，需要切换。
"""
import numpy as np
import sys
import struct
import os

# mnist_path = "./datasets/MNIST/"
def print_log(s):
    """
    对print函数的一种变相重写 防止在被其他文件调用时 输出不必要的调试信息
    """
    if __name__ =="__main__":
        print(s)
    else:
        pass
def loadMinistImage(filename):
    binfile = open(filename,'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>4i',buffers,0)#默认小端格式 需要变成大端格式则改成<
    #一个i就是32位整数表示4字节,offset=4表示偏移4个字节
    if head[0]!=2051:
        print_log("Errors occured in image file")
    else:
        print_log("Read image file succeed")
        img_num = head[1]
        img_width = head[2]
        img_height = head[3]
        bits = img_num*img_height*img_width
        #60000*28*28的字节
        bits_string=">"+str(bits)+"B"#以一个字节为单位的连续结构，而i是以4个字节为单位
        offset = struct.calcsize('4i')  # ��λ��data��ʼ��λ��
        imgs = struct.unpack_from(bits_string, buffers, offset)
        binfile.close()
        imgs = np.reshape(imgs,[img_num,img_width,img_height])
        #变成[60000 28 28]的矩阵
        return imgs
def loadMinistLable(filename):
    binfile = open(filename,'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>2i',buffers,0)#默认小端格式 要变成大端格式则改成<
    #一个i就是32位整数表示4字节,offset=4表示偏移4个字节
    if head[0]!=2049:
        print_log("Errors occured in label file")
    else:
        print_log("Read label file succeed")
        img_num = head[1]
        bits = img_num
        bits_string=">"+str(bits)+"B"#以一个字节为单位的连续结构，而i是以4个字节为单位
        offset = struct.calcsize('2i')  # 定位到data开始的位置
        labels = struct.unpack_from(bits_string, buffers, offset)
        binfile.close()
        labels = np.reshape(labels,[img_num])
        return labels
def load_data(mnist_path,get_new=True,normalization=False,one_hot=False,detype=np.float32):
    if get_new==True:
        train_images = loadMinistImage(os.path.join(mnist_path,"train-images.idx3-ubyte"))
        train_labels = loadMinistLable(os.path.join(mnist_path,"train-labels.idx1-ubyte"))
        test_images = loadMinistImage(os.path.join(mnist_path,"t10k-images.idx3-ubyte"))
        test_labels = loadMinistLable(os.path.join(mnist_path,"t10k-labels.idx1-ubyte"))
        np.savez(os.path.join(mnist_path,"mnist.npz"),k1=train_images,k2=train_labels,k3=test_images,k4=test_labels)
    else:
        npzfile=np.load(os.path.join(mnist_path,'mnist.npz'))
        train_images = npzfile['k1']
        train_labels = npzfile['k2']
        test_images = npzfile['k3']
        test_labels = npzfile['k4']

    if normalization==True:
        """
        to float64
        """
        train_images = train_images/255.0
        test_images = test_images/255.0
    else:
        train_images = train_images/1.0
        test_images = test_images/1.0

    if one_hot==True:
        new_train_labels = np.zeros(shape=(train_labels.shape[0],10))
        for i in range(train_labels.shape[0]):
            new_train_labels[i,train_labels[i]]=1.0
        new_test_labels = np.zeros(shape=(test_labels.shape[0],10))
        for i in range(test_labels.shape[0]):
            new_test_labels[i,train_labels[i]]=1.0
        return (train_images.astype(detype),new_train_labels.astype(detype)),(test_images.astype(detype),new_test_labels.astype(detype))
    else:
        return (train_images.astype(detype),train_labels.astype(detype)),(test_images.astype(detype),test_labels.astype(detype))
if __name__ == "__main__":
    (train_images,train_labels),(test_images,test_labels)=load_data(mnist_path=mnist_path,get_new=False,normalization=True,one_hot=True,detype=np.float64)
    print(train_images.shape,train_images.dtype)# (60000, 28, 28) float64
    print(train_labels.shape,train_labels.dtype)#
    print(test_images.shape,test_images.dtype)#
    print(test_labels.shape,test_labels.dtype)#
    (train_images,train_labels),(test_images,test_labels)=load_data(mnist_path=mnist_path,get_new=False,normalization=False,one_hot=False,detype=np.float32)
    print(train_images.shape,train_images.dtype)# (60000, 28, 28) float64
    print(train_labels.shape,train_labels.dtype)#
    print(test_images.shape,test_images.dtype)#
    print(test_labels.shape,test_labels.dtype)#
