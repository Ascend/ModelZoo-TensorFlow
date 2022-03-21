# Readme_NPU

### 1. 关于项目
本项目为复现 “Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network” 论文算法  

论文链接为： [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf)  

原作者开源代码链接为： [code](https://github.com/YadiraF/PRNet)  

该文章提出了一种能够同时完成3D人脸重建和稠密人脸对齐的端对端的方法–位置映射回归网络（PRN），效果示意图如下所示：  

![prnet](https://images.gitee.com/uploads/images/2021/0818/200336_92843d93_9227151.gif "prnet.gif")



### 2. 关于依赖库

见Requirements.txt， 需要安装 numpy>=1.14.3， scikit-image， scipy， opencv-python，dilb等库。



### 3. 关于数据集
模型训练使用300W_LP dataset数据集，数据集请用户自行获取。

数据集训练前需要做预处理操作，请用户参考GPU开源链接,将数据集封装为tfrecord格式。


其中各数据集均经过处理，直接解压即可。图片尺寸统一处理为256*256  

将训练集 AFW,  IBUG  解压后放入  /Dataset/TrainData/  中  

将测试集 LFPA,  AFLW2000  解压后放入  /Dataset/TestData/  中  

现已经上传百度云：https://pan.baidu.com/s/1yoKgL0NoKM7vXTeJhUj_8w#list/path=%2F （password：77d0 ）


### 4. 关于训练

将 BFM.mat以及BFM_UV.mat（https://pan.baidu.com/share/init?surl=Y_Zy2o2JEtZLfMIgT2QSVg ， password：s85o） 放入 face3d/examples/Data/BFM/Out 文件夹中  
将 weight_mask_final.jpg 放入 Data/uv-data 文件夹中

训练脚本为train.py。训练时在终端输入sh run_train.sh即可


### 5. NPU性能  
NPU（Ascend910）每个iteration耗时 0.27ms。每个epoch共1749个iteration，共耗时472ms  
GPU（使用华为弹性云服务器ECS）每个iteration耗时 0.28ms，每个epoch共1749个iteration，共耗时490ms


### 6. loss下降曲线：
![loss](https://images.gitee.com/uploads/images/2021/0803/195247_0b2dc43a_9227151.png "loss_npu.png")

### 7. 关于测试：
论文中测试集精度为：  
![论文测试集精度](https://images.gitee.com/uploads/images/2021/0818/204949_3b77ee8e_9227151.png "捕获.PNG")  

使用NPU训练后得到的测试集精度均优于原论文中精度，两者精度对比如下图所示：
![复现精度](https://images.gitee.com/uploads/images/2021/0819/135654_48a0b699_9227151.png "捕获.PNG")
