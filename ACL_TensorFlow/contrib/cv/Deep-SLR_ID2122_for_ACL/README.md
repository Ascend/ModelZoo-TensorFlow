## 数据需求

所需测试文件应为 raw data 数据，并需要与训练时相同的模板。测试文件的具体尺寸与训练时生成的模型中结点的尺寸有关。默认为 12 通道大小为 256 x 232 的raw data数据。

## 推理环境配置

推理环境搭建方法请参考以下教程。

- [ATC工具使用环境搭建](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)
- [msame工具为模型推理工具](https://gitee.com/ascend/tools/tree/master/msame)

## 精度指标

下表给出了论文中、GPU、NPU 训练和 NPU 推理的测试精度，均使用了示例数据。

|      |  论文精度   | GPU 精度 | NPU 训练精度 | **NPU 推理精度** |
| :--: | :---------: | :------: | :----------: | :--------------: |
| PSNR | 34.46±1.22  |  34.361  |    34.445    |    **34.443**    |
| SSIM | 0.958±0.011 |  0.921   |    0.924     |    **0.924**     |

可见，SSIM 精度指标与 NPU 训练精度持平；对于 PSNR 精度指标，由于 (34.445-34.443)/34.445=0.00005806357962<<0.01，可以认为推理精度与 NPU 训练精度保持一致，符合验收标准。

## 性能指标

平均推理性能：1708.78000 ms

## 推理流程

### 生成 om 模型

首先需要将 pb 模型文件转换为 om（**O**ffline **M**odel）文件。本项目的 pb 文件生成方式可以参考“ckpt2pb.py”脚本，它将使用 NPU 训练的 checkpoint 模型转换为 pb 模型。

下面是使用 atc 工具将 pb 模型转换为 om 文件的形式命令。

```shell
atc --model=deep-slr-model-100.pb --framework=3 --output=deep-slr-model-100 --soc_version=Ascend310 --log=info
```

其中，`--model` 指定 pb 模型路径，`--output` 指定生成 om 文件路径。

如果希望在转换时修改输入结点的尺寸，请附加 `-input_shape` 参数，如

```shell
atc --model=deep-slr-model-100.pb --framework=3 --output=deep-slr-model-100 --soc_version=Ascend310 --log=info --input_shape="atb:1,24,256,232,1;mask:1,12,256,232,1"
```

### 测试数据转换

生成 om 文件后，需要将所需重建图像数据（raw data）使用“convert.py”文件保存为二进制文件（.bin）。根据需求进行修改时，请参考代码中的注释。

### 执行模型并输入数据

下面是使用 msame 工具执行模型并输入数据的形式命令。

```shell
msame --model deep-slr-model-100.om --input "./input/atb.bin,./input/mask.bin" --output ./out
```

其中，`--model` 指定 om 路径，`--input` 指定输入结点的 .bin 文件路径，`--output` 指定输出文件的目标位置（示例中即为“out”文件夹下）。

## 测试输出结果精度

如果模型执行成功，则会在指定的输出路径生成二进制文件，此即图像的重建结果，同样为 raw data，因此需要使用“indexes.py”进行后处理并测试精度。使用时可能需要根据实际情况进行相应的修改。

在默认情况下，测试结果会保存在脚本文件所在目录下，为“test_res.txt”和“test_res.png”。其中“test_res.txt”中的后两个数据依次为测试结果的 PSNR 和 SSIM 指标。

## 样例文件

请在[此](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=VLLhxma8xiB2J1Z74pgqFBCqO9Pr/IJI6mQzxjaFzE2rll8Ls+jjTSync2VcLxvsU97qLyr/75lyRNkHlJkyShSh/RJBTFBAmtxZ2sWSpnlwRA6E5+CGVO4JfJbP3s5Hl3iOx3o7fPkS994jwroDWUO2kmEd4gO3KHQI5Rkqc0DdP/O+/f35WRmWgVoNq0tnE/+S70afet5bWRWsfmu/aZCX1bIIEfTPcD/iKyuSHvhUSz8NUty0+ptU/HiE58p3QkeS1ygtxI7h9iQZQT1F9XlKw9gPXy+AQJgwypfN+fek1n1bi30mT4YmTODPav0Fk9myhMQ1Fdz1NyyYpVSKLV9DTOojRuph/M/V/rFk4GPJARj0ou9XAu5v63r+9ayOxTGPvGEiMb7NCyJJuHUfgC3eBjcxiFEcAPm2QalsUzzOiy1EmRSC3GshCKsg9ow0lZBUSoOAUU11yXBjVLp3HD4Frq09Fx6r0XgYGMPC1M32PNGzSKjzprfREGXHyyLzME5IE5GRWo75dp/ag4XWqRNlKCMFX42DL2p/MA6dcrP9r2McKVX2BUQMgNgGJpuU)（提取码：666666，有效期至：2022/12/03 20:37:57 GMT+08:00）下载，其中提供了转换所需的 pb 文件，测试数据，生成的 om 文件及测试结果 bin 文件。

其中，需要和提供的 Python 脚本配合使用的文件有“tst_img.npy”，“verdenmask_6f.mat”和“deep-slr-model-100_output_0.bin”。前两个文件分别是测试图像数据和模板文件，用于生成输入数据“atb.bin”和“mask.bin”；最后一个文件是成功执行 om 之后生成的图像重建结果文件，通过提供的 Python 脚本可以验证图像重建结果。
