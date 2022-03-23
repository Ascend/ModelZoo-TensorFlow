# 概述
--NGNN对服装的兼容性进行评估，研究时尚推荐的实际问题。
* 参考论文
https://arxiv.org/abs/1902.08009
* 训参考训练相关
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/graph/NGNN_ID1246_for_TensorFlow

# ATC命令
`atc --model=/home/HwHiAiUser/AscendProjects/NGNN/frozen_model_acc.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/NGNN/ngnn_acc_6input --soc_version=Ascend310   --input_shape="Placeholder:16,120,2048;Placeholder_1:16,120,2048;Placeholder_2:16,120,2757;Placeholder_3:16,120,2757;Placeholder_4:16,120,120; Placeholder_5:16,120,120" --log=info --out_nodes="s_pos_output:0"` `

# ATC+autotune
`atc --model=/home/HwHiAiUser/AscendProjects/NGNN/frozen_model_acc.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/NGNN/ngnn_acc_autotune --soc_version=Ascend310 --log=info --out_nodes="s_pos_output:0" --auto_tune_mode="RL,GA"`

# ckpt模型+PB模型+OM模型+较大的数据集
链接：https://pan.baidu.com/s/12q1Kien31nWtb_V-CPrISA 
提取码：b0iv 

# 数据集准备
 `python toBin.py`

# 使用msame工具推理
`"/home/HwHiAiUser/AscendProjects/NGNN/data/acc_image,/home/HwHiAiUser/AscendProjects/NGNN/data/acc_image,/home/HwHiAiUser/AscendProjects/NGNN/data/acc_text,/home/HwHiAiUser/AscendProjects/NGNN/data/acc_text,/home/HwHiAiUser/AscendProjects/NGNN/data/acc_graph,/home/HwHiAiUser/AscendProjects/NGNN/data/acc_graph" --output "/home/HwHiAiUser/AscendProjects/NGNN/output/" --outfmt TXT`

# 精度测试
```
python acc.py
python auc.py
```
| Accuracy    | AUC         |
|-------------|-------------|
| 0.833147942 | 0.971449759 |
