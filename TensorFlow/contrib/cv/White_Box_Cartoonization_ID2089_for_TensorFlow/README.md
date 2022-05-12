- [基本信息](#基本信息.md)

- [概述](#概述.md)

- [训练环境准备](#训练环境准备.md)

- [快速上手](#快速上手.md)

- [训练结果](#训练结果.md)

- [高级参考](#高级参考.md)
  <h2 id="基本信息.md">基本信息</h2>
  
  **发布者（Publisher）：Huawei**
  
  **应用领域（Application Domain）：Natural Language Processing** 
  
  **版本（Version）：1.2**
  
  **修改时间（Modified） ：2022.4.24**
  
  **大小（Size）：509.3MB**
  
  **框架（Framework）：TensorFlow 1.12.0**
  
  **模型格式（Model Format）：ckpt**
  
  **精度（Precision）：Mixed**
  
  **处理器（Processor）：昇腾910**
  
  **应用级别（Categories）：Official**
  
  **描述（Description）：基于TensorFlow框架对图片进行动漫化处理** 
  
  <h2 id="概述.md">概述</h2>
  
      White-box Cartoonization是2020年由Xinrui Wang 和 Jinze Yu提出的对图片进行动漫化处理的算法，刊登在IEEE Conference on Computer Vision and Pattern Recognition上。
  从图像中分别识别三种白盒表示：包含卡通图像平滑表面的表面表示，指赛璐珞风格工作流中稀疏色块和扁平化全局内容的结构表示，以及反映卡通图像中高频纹理、轮廓和细节的纹理表示。生成性对抗网络（GAN）框架用于学习提取的表示并对图像进行自动化。
  
  - 参考论文：
    
      https://github.com/SystemErrorWang/White-box-Cartoonization/tree/master/paper
    
  - 参考实现：
    
      https://github.com/SystemErrorWang/White-box-Cartoonization 
    
  - 适配昇腾 AI 处理器的实现：
        
            
      https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/nlp/LeNet_for_TensorFlow
            
  
  
    - 通过Git获取对应commit\_id的代码方法如下：
      
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```
  
    ## 默认配置
  
    - 训练数据集：
    
      风景照片与人物照片以及对应的风景卡通照与人物卡通照
      
    图片输入格式：jpg
    
    - 测试数据集预处理（以MNIST验证集为例，仅作为用户参考示例）
    
      测试照片
      
    图像输入格式：jpg
    
    - 训练超参
    
      - Total iteration: 100000
  
  
    ## 支持特性
  
    | 特性列表  | 是否支持 |
    |-------|------|
    | 分布式训练 | 否    |
    | 混合精度  | 是    |
    | 并行数据  | 是    |
  
    ## 强制精度训练
  
    昇腾910 AI处理器提供强制精度功能，通过算子溢出检测我们发现Conv2d算子存在溢出的情况。由于在NPU上该算子仅支持fp16精度，因此我们需要开启强制精度。
  
    ## 开启强制精度
  
    脚本已默认开启强制置精度precision_mode参数的脚本参考如下。
  
      ```
      custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
      custom_op.name = "NpuOptimizer"
      config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
      config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
      custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
      ```
  
    ## Loss_Scale
  
    由于算子精度溢出，我们需要开启Loss_Scale来进行精度的优化。由于代码中有两个优化器，因此我们开启两个loss_scale_manager。
  
      ```
      loss_scale_manager_g = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,
                                                                   decr_every_n_nan_or_inf=2, decr_ratio=0.5)
      loss_scale_manager_d = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,
                                                                   decr_every_n_nan_or_inf=2, decr_ratio=0.5)
      ```
  
  在优化器上添加loss_scale.
  
  ```
  g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)
  g_optim = NPULossScaleOptimizer(g_optim, loss_scale_manager_g).minimize(g_loss_total, var_list=gene_vars)
  
  d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)
  d_optim = NPULossScaleOptimizer(d_optim, loss_scale_manager_d).minimize(d_loss_total, var_list=disc_vars)
  ```
  
  获取loss_scale并打印
  
  ```
  lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
  l_s_g, _, g_loss, r_loss = sess.run([lossScale,g_optim,g_loss_total, recon_loss],
                                      feed_dict={input_photo: photo_batch, input_superpixel: superpixel_batch,
                                                 input_cartoon: cartoon_batch})
  
  
  l_s_d, _, d_loss, train_info = sess.run([lossScale, d_optim, d_loss_total, summary_op],
                                           feed_dict={input_photo: photo_batch, 
                                                      input_superpixel: superpixel_batch,
                                                      input_cartoon: cartoon_batch})
  print('Iter: {}, loss_scale g: {}, loss_scale d: {}'.format(total_iter, l_s_g, l_s_d))
  ```
  
    <h2 id="训练环境准备.md">训练环境准备</h2>
  
    1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
    2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。
    
        当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。
    
        **表 1** 镜像列表
    
        <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
        </td>
        <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
        </td>
        </tr>
        </tbody>
        </table>
  
  
    <h2 id="快速上手.md">快速上手</h2>
  
    - 数据集准备
    1. 模型训练使用MNIST数据集，数据集请用户自行获取。
  
    ## 模型训练<a name="section715881518135"></a>
  
    - 单击“立即下载”，并选择合适的下载方式下载源码包。
    
    - 启动训练之前，首先要配置程序运行相关环境变量。
    
      环境变量配置信息参见：
    
         [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    - 单卡训练 
    
      1. 配置训练参数。
      
         首先在脚本npu_train.sh中，配置code_dir, word_dir, dataset_path, output_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。
      
         ```
          total_iter=100000
          data_path="../dataset"
         ```
         
      2. 启动训练。
      
         启动单卡训练 （脚本为LeNet_for_TensorFlow/test/train_full_1p.sh） 
      
         ```
       bash train_full_1p.sh --data_path=../MNIST
         ```
  
    <h2 id="训练结果.md">训练结果</h2>
  
    - 精度结果比对
  
  |精度指标项|NPU实测|GPU实测|
  |---|---|---|
  |Fid to photo|29.13|28.79|
  
  
  
  | 精度指标       | NPU实测 | GPU实测 |
  | -------------- | ------- | ------- |
  | Fid to cartoon | 107.41  | 101.31  |
  
    - 性能结果比对  
  
  |性能指标项|NPU实测|GPU实测|
  |---|---|---|
  |FPS|1.46it/s|1.06it/s|
  
  
    <h2 id="高级参考.md">高级参考</h2>
  
    ## 脚本和示例代码
  
    ```
    ├── README.md																	//代码说明文档
    ├── modelarts_entry.py                        //modelarts平台开启训练文件
    ├── modelarts_entry_acc.py										//测试精度文件
    ├── modelarts_entry_perf.py                   //测试性能文件
    ├── modelzoo_level.txt
    ├── npu_train.sh
    ├── requirements.txt														//训练依赖列表
    ├── test                                     		//测试脚本
    │   ├── train_full_1p.sh
    │   └── train_performance_1p.sh
    ├── test_code																		//测试文件夹
    │   ├── cartoonize.py													  //测试文件（生成动漫图片）
    │   ├── guided_filter.py
    │   ├── network.py
    │   ├── saved_models
    │   └── test_images
    ├── train_code																	//训练代码列表
    │   ├── guided_filter.py
    │   ├── layers.py
    │   ├── loss.py
    │   ├── network.py
    │   ├── ops_info.json
    │   ├── pretrain.py
    │   ├── selective_search
    │   ├── train.py                                 //训练文件
    │   └── utils.py
    └── vgg19_no_fc.npy                              //预训练文件
    ```
  
    ## 脚本参数
  
    ```
    --data_path              数据集路径，默认：cache/dataset
    --output_path.           输出路径，默认： cache/output
    --batch_size             每个NPU的batch size，默认：16
    --total_iter             迭代次数，默认：100000
    ```
  
    ## 训练过程
  
    1.  通过“模型训练”中的训练指令启动单卡卡训练。
    
    2.  参考脚本的模型存储路径为./output/train_cartoon/saved_models/
