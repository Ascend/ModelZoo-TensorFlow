
<h2 id="概述.md">概述</h2>

This is the project about our JSHDR (CVPR2021). Our paper is here A Multi-Task Network for Joint Specular Highlight Detection and Removal.

    ```
    @InProceedings{fu-2021-multi-task,
    author = {Fu, Gang and Zhang, Qing and Zhu, Lei and Li, Ping and Xiao, Chunxia},
    title = {A Multi-Task Network for Joint Specular Highlight Detection and Removal},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021},
    pages = {7752-7761},
    month = {June},
    tags = {CVPR},
    }
    ```

<h2 id="om模型转换.md">om模型转换</h2>
在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:

    ```
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    atc --model=/home/HwHiAiUser/AscendProjects/jshdr/best_model2.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/jshdr/best_model --soc_version=Ascend310 --input_shape="input_1:1,64, 64, 3;input_2:1,64, 64, 3"
    ```

请从此处下[pb模型](https://jshdr-pb-om.obs.cn-north-4.myhuaweicloud.com:443/best_model_10.pb?AccessKeyId=IXCI6DV92LIG7HESD9I0&Expires=1657795521&Signature=4uXE6A6NmhOmuz94BMdOOnsOkxE%3D) \
请从此处下[om模型](https://jshdr-pb-om.obs.cn-north-4.myhuaweicloud.com:443/best_model10.om?AccessKeyId=IXCI6DV92LIG7HESD9I0&Expires=1657795543&Signature=Tgzq6U0ATxNDQEDH/MwV2BFuXfo%3D)
<h2 id="使用msame工具推理.md">使用msame工具推理</h2>

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

<h3>1.数据转换bin</h3>

修改img2bin.py文件中的图片地址，改成需要测试的图片地址

执行img2bin.py文件将需要测试的图片转换为推理需要的——>bin数据

<h3>2.推理</h3>
使用msame推理工具，发起推理测试，推理命令如下：

    ‘’‘
    cd /home/msame/out
    ./msame --model "/home/HwHiAiUser/AscendProjects/jshdr/best_model10.om" --input "/home/msame/out/4.bin" --output "/home/msame/out/" --outfmt PNG --loop 1
    ’‘’

<h3>3.推理结果</h3>

    ‘’‘
    [INFO] acl init success
    [INFO] open device 0 success
    [INFO] create context success
    [INFO] create stream success
    [INFO] get run mode success
    [INFO] load model /home/HwHiAiUser/AscendProjects/jshdr/best_model10.om success
    [INFO] create model description success
    [INFO] get input dynamic gear count success
    [INFO] create model output success
    /home/msame/out//20220115_185917
    [INFO] start to process file:/home/msame/out/4.bin
    [INFO] model execute success
    Inference time: 11.713ms
    [INFO] get max dynamic batch size success
    [INFO] output data success
    Inference average time: 11.713000 ms
    [INFO] destroy model input success
    [INFO] unload model success, model Id is 1
    [INFO] Execute sample success
    [INFO] end to destroy stream
    [INFO] end to destroy context
    [INFO] end to reset device is 0
    [INFO] end to finalize acl

    ’‘’

<h2 id="4。查看测试图片的输出结果.md">4。查看测试图片的输出结果</h2>

打开推理生成的txt文件，即可以得到om模型所输出的图片。（需要bin2img转换显示）


