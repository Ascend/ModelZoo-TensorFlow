# 原始模型
在训练得到ckpt数据之后，已经使用tensorflow转换成名为srnet.pb的pb文件，通过[此链接](https://cann-nju-srnet.obs.myhuaweicloud.com:443/for_acl/srnet.pb?AccessKeyId=DNOFMBDXF3DTPNGYLYN7&Expires=1657153689&Signature=ME/qgg6WBtJdYDceSXQ3VDXa/SE%3D)进行下载，为了后续工作的正常执行，请将下载的pb文件存放在当前目录下名叫model的文件夹之中。
# 转OM模型
为了在ascend平台上进行推理，需要将pb模型转换成om模型，转换使用ascend atc工具完成，示例命令参考如下(atc.sh)：  
```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    atc \
    --framework=3 \
    --model=./model/srnet.pb \
    --output=./model/srnet \
    --input_format=NHWC \
    --input_shape="input_t:1,64,128,3;input_s:1,64,128,3" \
    --out_nodes="o_f:0" \
    --soc_version=Ascend310
```
# 编译MSAME推理工具
在ascend平台上的推理通过msame推理工具推理om模型进行，msame推理工具的下载与编译请参考[此链接](https://gitee.com/ascend/tools/tree/master/msame) 。
# 性能测试
在完成msame推理工具的下载和编译之后，可以参考如下命令(msame.sh)进行性能测试：
```bash
    ./msame --model ./model/srnet.om --output ./output/ --loop 100
```
# 精度&性能自测试用例
训练数据已经转换成bin类型，使用训练集的[前100组数据](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=EVlvCz2hJFIZnLCeJVgI95BA+VZFXQ/LMObUbAZkl/4J6CDJWhGZTnnZa8R7v5rcr8LrlSFrRG4dmX7gTpkdAA5JHK9y158dOpN7+ZDQYsGX2XnQSaoVMlZhr651UoXeH2YKInu/fCM9nvl6i9clAZJ4HIjBI4sfILlMbMxx6jzfi4qXlJ0vmETEoY2R1AArVJHxO3WKiHwkenvBS2MtJ/yFpzJqJAgAEjpt1jJ/zJlnb2vPpVXUk/l1pJeGHUDjgB7Ch7Tyfx2WNyyB6/yjEzsq/8/BJyRnwBf7UYq9p6nrjpwvkXLAtUbvMwI6eTsVh8lUXrYaTRO76l1/aVPFcBfRdpAvGydotpd+Bkz1pMlVMiR82wFbU+BTGb/cnMycco/tmKXqBr78qWsY3FwtS/dRD8TpKHEFsgo6gNqux60Kb96f33dsz0TywgMTBeKkEYY1wZDNZrwXKdNVWSXLKi9aeCdSD+Z4M7MdEj2RRWFC/DIFaXzZfr1uAo4qKlacF3xciPc3yw3gVwfJJsydxzqdAPkFMHS97Mf885rk/SLslPSN6t9BAIoZYjWAsNRCb29wpizd7L3fHipPjEBJ5w==)（提取码：112233）来验证，请将数据下载到当前文件夹下的bindata目录之中。
验证过程通过执行当前目录下的entrance.py来进行，示例命令如下：
```python
python ./entrance.py
```
# 自定义数据预测
用户可以使用自己的数据进行预测，流程如下：
1. 提供原始的场景文本图像(I_S)和格式化的字体图像(I_T)并调用img2bin.py将图像转换成bin文件
2. 使用msame进行推理，得到推理结果bin文件
3. 使用bin2img.py将推理结果结果bin文件转换回图像
用户也可以将准备好的I_S和I_T分别存放在./predict/i_s以及./predict/i_t目录下并保持对应图片的名称一致，然后调用customized_predict.sh进行预测，预测结果分别保存在./predict/output_bin以及./predict/output_img目录下，名称同对应的源文件保持一致。
