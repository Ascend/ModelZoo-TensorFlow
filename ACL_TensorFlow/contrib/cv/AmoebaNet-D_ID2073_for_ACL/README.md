<h2 id="概述.md">概述</h2>

AmoebaNet-D是由AmoebaNet演化神经架构搜索算法搜索出的一个图像分类神经网络，本项目用于该模型的离线推理。

- 参考论文：

    [Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019, July). Regularized evolution for image classifier architecture search. In Proceedings of the aaai conference on artificial intelligence (Vol. 33, No. 01, pp. 4780-4789).](https://arxiv.org/pdf/1802.01548.pdf) 


- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/zero167/ModelZoo-TensorFlow/edit/master/ACL_TensorFlow/contrib/cv/AmoebaNet-D_ID2073_for_ACL]
(https://gitee.com/zero167/ModelZoo-TensorFlow/edit/master/ACL_TensorFlow/contrib/cv/AmoebaNet-D_ID2073_for_ACL)      



## 模型固化<a name="section168064817164"></a>

- 直接获取

直接下载获取，固化模型obs链接：obs://amoebanet/test_299.pb

- 训练获取

  1. 从头训练或使用已有的checkpoint运行训练脚本将模型保存为saved_model格式文件（需要在amoeba_net.py脚本中修改保存路径）。
       python amoeba_net.py

  2. 将saved_model格式文件冻结为pb文件（需要在freeze.py文件中修改路径）。
       python freeze.py


## 使用ATC工具将pb文件转为离线模型<a name="section20779114113713"></a>

命令行示例：
    atc --model=/home/test_user03/tpu-3/models/official/amoeba_net/test_299.pb --framework=3 --output=/home/test_user03/tpu-3/models/official/amoeba_net/test_om --soc_version=Ascend310 --input_shape="input_ids:64,299,299,3" --precision_mode=allow_fp32_to_fp16




## 制作验证集bin文件<a name="section168064817164"></a>

运行data_make.py脚本（需要修改data_make.py脚本中的路径）。

 python data_make.py

## 使用msame工具进行离线推理获取输出bin文件<a name="section168064817164"></a>

命令行示例：
    ./msame --model  "/home/test_user03/tpu-3/models/official/amoeba_net/test_om.om" --output "/home/test_user03/tools/msame/out" --input "/home/test_user03/tpu-3/models/official/amoeba_net/bin"  --outfmt bin


## 使用输出bin文件验证推理精度<a name="section168064817164"></a>

运行inference.py脚本（需要修改inference.py脚本中的路径）
 python inference.py

参考top1精度：76.12%


