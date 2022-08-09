# 简介
coltran模型是论文“Colorization Transformer”的Tensorflow实现（基于晟腾NPU）。该模型包含三个子网络：`colorizer`, `color_upsampler` 以及 `spatial_upsampler`.
* colorizer是一种自回归、基于自注意力的架构，由条件转换器层组成。 它对低分辨率 64x64 灰度图像逐像素粗略着色。
* color_upsampler是一个并行的、确定性的基于自注意力的网络。 它将粗糙的低分辨率图像细化为 64x64 RGB 图像。
* spatial_upsampler的架构类似于color_upsampler。 它将低分辨率 RGB 图像超分辨为最终输出。
# 文件说明
* models：网络定义
* configs：各子网络训练参数
* test_configs：各子网络测试参数
* utils: 初始化依赖相关函数
* train.py: 训练脚本
* dataset.py：数据处理脚本
# 依赖安装
* pip install tensorflow-gpu==1.15.0
* pip install tensorflow_datasets==3.2.1
* pip install ml_collections
# 数据集
* imagenet2012
* obs地址：`obs://imagenet2012-lp/coltran_re/imagenet2012/`
# GPU复现训练
* obs地址：`obs://imagenet2012-lp/coltran_re/`
* colorizer训练log及ckpt模型：`obs://imagenet2012-lp/coltran_re/coltran_colorizer/`
* color_upsampler训练log及ckpt模型：`obs://imagenet2012-lp/coltran_re/coltran_color_upsampler/`
* spatial_upsampler训练log及ckpt模型：`obs://imagenet2012-lp/coltran_re/coltran_spatial_upsampler/`
# NPU复现训练
* colorizer训练log及ckpt模型：`obs://imagenet2012-lp/coltran_log/MA-new-coltran_modelarts-05-17-17-36/`
* color_upsampler训练log及ckpt模型：`obs://imagenet2012-lp/coltran_log/MA-new-coltran_modelarts-05-19-13-49/`
* spatial_upsampler训练log及ckpt模型：`obs://imagenet2012-lp/coltran_log/MA-new-coltran_modelarts-05-22-09-13/`
# 训练精度对比
|  |                |  GPU | NPU  | 
|-----|-----------|------|--------|
|colorizer | loss| ~0.58 | ~0.58 | 
|color_upsampler | loss | ~2.88 | ~2.80 | 
|spatial_upsampler | loss | ~2.66 | ~2.63 | 
# 训练性能对比
|  |                |  GPU | NPU  | 
|-----|-----------|------|--------|
|colorizer | loss| 7.8 examples/s | 27.4 examples/s | 
|color_upsampler | loss | 22.6 examples/s | 57.4 examples/s | 
|spatial_upsampler | loss | 1.6 examples/s | 4 examples/s | 
# 测试精度
* TODO