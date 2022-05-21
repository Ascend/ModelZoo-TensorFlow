# 基本信息：
发布者（Publisher）：Huawei  
应用领域（Application Domain）： Signature-detection
版本（Version）：1.0  
框架（Framework）：TensorFlow 1.15.0  
模型格式（Model Format）：ckpt  
处理器（Processor）：昇腾910  
应用级别（Categories）：Research  
描述（Description）：基于TensorFlow框架的特征点检测网络训练代码  

# 概述：
Factorized是一个通过分解空间嵌入对对象地标进行无监督学习
参考论文及源代码地址：  
Unsupervised learning of object landmarks by factorized spatial embeddings
https://github.com/alldbi/Factorized-Spatial-Embeddings


#推理过程

1.执行ckpt2pb.py 生成pb文件
ckpt = './V0139/-625000'# ckpt模型路径
pb = './pb/new3.pb' # pb模型生成路径

2.pb模型添加placehoder
执行pbdeal.py
将pb文件改为txt后在最开始手动改为Placeholder节点
修改后再转回.pb文件
例子 old.pb 改为new3.pb


3.pb转om
atc --model=/home/new3.pb --framework=3 --output=new0 --soc_version=Ascend310 -- out_nodes="cnn_tower/layer_6/BatchNorm/Relu"

4.pic转bin
执行pre.py
生成 1.bin 2.bin deformationlog.npy

5.推理
./msame --model /home/new0.om --input /home/1.bin --output /home/outbin1 --outfmt TXT --loop 1
./msame --model /home/new0.om --input /home/2.bin --output /home/outbin2 --outfmt TXT --loop 1

6.得出输出结果 做后处理 取精度
执行enddeal.py 
参数 
new0_output_0.txt --推理1.bin生成
new0_output_1.txt --推理2.bin生成
deformationlog.npy --转bin生成

deformationlog = np.load("deformationlog.npy")
predA = np.loadtxt("new0_output_0.txt")
predB = np.loadtxt("new0_output_1.txt")

#推理模型下载

数据集
obs://fse-1/datapic/

bin获取
obs://fse-1/bin/

ckpt模型路径
obs://fse-1/workplace/MA-new-Factorized-Spatial-Embeddings-master-04-25-12-53/code/output/

pb模型路径
obs://fse-1/pb/

om模型路径
obs://fse-1/om/

enddeal.py 所需参数
obs://fse-1/enddeal/

#推理模型精度性能
log1.png 
log2.png

精度性能信息对比：
gpu：
精度loss_align 4-8
性能image/sec 196.5
npu
精度loss_align 4-8
性能image/sec 233.1 
推理
精度loss_align 0.789
性能image/sec 245.7
