# 简介
CS_GAN模型是论文“Deep Compressed Sensing”的Tensorflow实现（基于晟腾NPU）。
# 文件说明
* models：网络定义
* configs：各子网络训练参数
* test_configs：各子网络测试参数
* utils: 初始化依赖相关函数
* main.py: 训练脚本
* net.py：定义网络
* gan.py：定义GAN模型
# 依赖安装
* pip install tensorflow-gpu==1.15.0
* pip install tensorflow_datasets==3.2.1
* pip install ml_collections
# 数据集
* cifar
* obs地址：`obs://csgan-npu/CS_GAN_ID2064/cifar/`
# GPU复现训练
* obs地址：`obs://csgan/`
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

# 测试精度
* TODO