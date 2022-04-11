###模型功能

利用卷积神经网络来完成时序问题的建模，提出的时序卷积网络（Temporal convolutional network， TCN）与多种RNN结构相对比，发现在多种任务上TCN都能达到甚至超过RNN模型。

论文链接：https://arxiv.org/pdf/1803.01271.pdf

开源代码链接：https://gitee.com/YuanTingHsieh/TF_TCN

npu实现pr链接：https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/TCN_ID1231_for_TensorFlow

###

### ckpt转pb

在华为云的Ascend310服务器中，toolkit中包含有freeze_graph.py文件，可以将ckpt转换为pb文件，可参考如下命令：

python3.7 /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py \
--input_checkpoint=./save.ckpt-50 \
--output_graph=./TCN.pb \
--output_node_names="ArgMax" \
--input_meta_graph=./save.ckpt-50.meta \
--input_binary=true  


###pb转om

使用ATC模型转换工具进行模型转换，可参考如下命令：

atc --model=/home/HwHiAiUser/AscendProjects/NGNN/pb_model/TCN.pb \
--framework=3 \
--output=/home/HwHiAiUser/AscendProjects/NGNN/TCN2 \
--soc_version=Ascend310 \
--input_shape="Placeholder_1:32,1020,1"\
--log=info\
--out_nodes="ArgMax"\

###使用msame工具推理

数据准备：
utils.py  中的data_generator函数生成数据序列，及数据集。
制作数据输入bin文件
命令：python data_bin.py ，输出得到test.bin文件，作为离线推理的输入

推理：
工具为命令行的运行方式，推理单个bin文件参考如下：
./msame --model /home/HwHiAiUser/AscendProjects/NGNN/modle3.om --input /home/HwHiAiUser/AscendProjects/inference/test.bin  --output /home/HwHiAiUser/AscendProjects/inference/out/output1 --outfmt TXT --loop 2
推理文件夹下的所有bin文件参考如下：
./msame --model /home/HwHiAiUser/AscendProjects/NGNN/modle3.om --input /home/HwHiAiUser/AscendProjects/inference/  --output /home/HwHiAiUser/AscendProjects/inference/out/output1 --outfmt TXT --loop 2

推理结果如图inference.png所示：
Inference average time: 61.484000 ms
Inference average time without first time: 3.077000 ms

###性能&精度

 性能

| GPU         | Ascend 910  |
| :--------- | :---------- |
| 0.5ms/iter | 3ms/iter |

#### 训练精度

| GPU      | Ascend 910 |
| :------- | :--------- |
| 100%   | 100%     |
