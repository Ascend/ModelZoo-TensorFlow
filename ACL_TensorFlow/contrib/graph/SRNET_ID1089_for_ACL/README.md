# 原始模型
在训练得到ckpt数据之后，已经使用tensorflow转换成名为srnet.pb的pb文件，通过[此链接](https://cann-nju-srnet.obs.cn-north-4.myhuaweicloud.com:443/pb/srnet.pb?AccessKeyId=DNOFMBDXF3DTPNGYLYN7&Expires=1655729803&Signature=2cfLSaKnfHsNQ41h0hSbrpeFPMk%3D)
进行下载，为了后续工作的正常执行，请将下载的pb文件存放在当前目录下名叫model的文件夹之中。
# 转OM模型
为了在ascend平台上进行推理，需要将pb模型转换成om模型，转换使用ascend atc工具完成，示例命令参考如下(atc.sh)：  
```bash
   atc --framework=3 --model=./model/srnet.pb --output=./model/srnet.om --input_format=NHWC --input_shape="input_t:1,64,128,3;input_s:1,64,128,3" --soc_version=Ascend910A
```
# 编译MSAME推理工具
在ascend平台上的推理通过msame推理工具推理om模型进行，msame推理工具的下载与编译请参考[此链接](https://gitee.com/ascend/tools/tree/master/msame) 。
# 性能测试
在完成msame推理工具的下载和编译之后，可以参考如下命令(msame.sh)进行性能测试：
```bash
    ./msame --model ./model/srnet.om --output ./output/ --loop 100
```
# 精度&性能自测试用例
训练数据已经转换成bin类型，使用训练集的[前100组数据](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=f7Ng+gPcaXow4C39826EjWn8nToI5cOakvcdO98a6cZPAMVs7zE12Ax4EmGOx3CMWEoGlMNtZOHtikQg/3iWzVPRgNK4WGf9bpiIAkbT0iVHPOEqXqkUcusmzL7GPNiCwNCtbR7asCfTVF+OteA0Za46y8zXz9Cn46eXP2A+nRjlvvbY6sBNO4mEuk64qK+Kt/iBbqFhE0Cn+0t8FvlbSsywCSRklmzoFikZWu9opKec9VonbnUm6nZk1GH/Q33eTt+g84jfUaC1VipprKze3tDGs4+Y1uHEtDbWnlS5dvjgpf0JQ1u+4UJr/LVDhgPuMaw2RvvNbz/DFcBR9GL1Vrt5vC0YDzgGG6kFDhEBrBCXcaC53cb9dTTYs4Lz36UeXmC/zvZ4Wgyh/C3jY8JdSxcvcnuM/hAAGEOV360D9aYW8xwQWghS9Lzo4W9TuqkmlsRVY2T63SYvNP2WP1txV+dXVrf+UKQdIsuTv769S2b5YZZu8KxVnbStMmZAH/vBk1l4Abk5SvElZNb0ja7h+1VfrxGgxs/dtFA+vfvlOXsdbxBL4yi8iy/1S6DYtOCp)（提取码337628）来验证，请将数据下载到当前文件夹下的bindata目录之中。
验证过程通过执行当前目录下的offline_inference_testcase.sh来进行，示例命令如下：
```bash
./offline_inference_testcase.sh
```