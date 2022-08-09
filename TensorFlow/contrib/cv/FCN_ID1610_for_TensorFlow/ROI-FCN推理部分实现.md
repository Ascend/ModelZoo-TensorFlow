# ROI-FCN推理部分实现
FCN推理部分实现，模型概述详情请看[FCN_ID1610_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/pulls/580)  README.md

## 训练环境

* TensorFlow 1.15.0
* Python 3.7.0

## 代码及路径解释

```

├── 20220804_195204
│   └── code1_output_0.bin  //om 预测输出结果
├── code1.om  //pb生成om文件
├── Data_Reader.py  
├── Data_Zoo1
│   └── Materials_In_Vessels //测试
├── images1.bin //输入bin文件
├── IOU.py
├── pbtest.py //pb测试
├── roimaps1.bin //输入bin文件
├── Test2.pb
└── tst.py //bin 测试
            
```


## 数据集
* Materials_In_Vessels

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WMagRWTzyCkm9a+cds5v8CNWaFudJLd6YGJldkGPRhiIWRwDEZx0E7vmGHVNGKMKTm/+JivX8K0mc51MWV+TgW4mk8notGKjVAeqzL+szBQKJcrpRiFawGne8mlP3ZZ7hCqg6CRYs8A19+C0eDx45lRvAsXvYxurqPvAlP1ISOHQYHFR0p5XK5z+qKNlBhiI6DGBtQjZ0nx9HEi1C5BIXb1zwwzRlkPGFidm3hRB62ag24QPWmsXdfpdSbn8vCX2Zng3CHEHvxNrR277WQFmqelnaA8jyTxfe+qwF4YOTqvxw58RmsaoDHNHm/M3I9ksmhsfmFC6xF/+WMEHWpYeTIGnfgDS/EDOnd++LdpWJfAX8e9lGpF8lkgruE8RfDrDsHbeKmEjBi7I3jgp1Dcczg2D9mIkn4Dtx3MyYnmJtiVdaF1D/c5aCORiS4OcqiQm2s8yjhBTYJIF4lle7sMl7un+3VQZLZXAznJPTOsOjrU7nqptD/zNsRJyb7an+oMNzXnwDMTvJSlDyv8Uo6RLeA==

提取码:
000000

*有效期至: 2023/07/31 16:27:22 GMT+08:00

## 模型文件
包括初始h5文件，固化pb文件，以及推理om文件
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WMagRWTzyCkm9a+cds5v8CNWaFudJLd6YGJldkGPRhiIWRwDEZx0E7vmGHVNGKMKTm/+JivX8K0mc51MWV+TgW4mk8notGKjVAeqzL+szBQKJcrpRiFawGne8mlP3ZZ7hCqg6CRYs8A19+C0eDx45lRvAsXvYxurqPvAlP1ISOHQYHFR0p5XK5z+qKNlBhiI6DGBtQjZ0nx9HEi1C5BIXd13FwsqqEeJjb+2J6g1QOmbZSisCnGZniRbbQ2OpvsIPWCT3XLQpZqEMQOUiSskegdgfzn+Tv2/e2wgVUm3YLGVUNFWTN13zbi6VzUlHBMynRH6x5PtUU8XaAdYFlkXwfphC0TibXulh+5RROzwtYjhDHioNCF1YIggX7VDTJa4C5Mc3WzivFjtHV44JCcbGvhYHcWo0Ktenv/x/6zfyovvAzk0SKOTCeCjFT27XFOgVUGzwJEJCFc71E6W+SxCYudBeGpneN9op9zA2DwIps0=

提取码:
000000

*有效期至: 2023/07/31 16:27:02 GMT+08:00

## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 :
```shell
atc --model=Test2.pb --framework=3 --output=code348 --soc_version=Ascend310 --out_nodes="Pred:0" --input_shape "input_image:4,480,854,3;ROIMap:4,480,854,1"

```
具体参数使用方法请查看官方文档。

## 使用msame工具推理

使用msame工具进行推理时可参考如下指令 msame.sh
```shell
./msame --model "/home/test_user03/code/code348.om" --input "/home/test_user03/code/images300.bin,/home/test_user03/code/roimaps300.bin" --output "/home/test_user03/code/" --outfmt BIN--loop 1
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 使用推理得到的bin文件进行推理
```shell
python3.7.5 tst.py
```

## 精度
* pb
* ![ad15db7fb1dd11fe63435884d28f67d](C:\Users\ADMINI~1\AppData\Local\Temp\WeChat Files\ad15db7fb1dd11fe63435884d28f67d.png)
* om

![b5751546d39cad46fb88c25f13b7215](C:\Users\ADMINI~1\AppData\Local\Temp\WeChat Files\b5751546d39cad46fb88c25f13b7215.png)

