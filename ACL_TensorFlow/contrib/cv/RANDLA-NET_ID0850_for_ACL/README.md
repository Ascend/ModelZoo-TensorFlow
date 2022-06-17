## 推理/验证过程<a name="section1465595372416"></a>

1. ckpt文件转pb文件，取6-fold-cross-validation中第一次训练保存的模型进行操作，此时的训练集是Area2-Area6,测试集是Area1。由于原始模型的数据传入接口为dataset，需要将它转换成placeholder的接口，具体实现参照dataset_to_placeholder.py，再参照ckpt_to_pb文件将ckpt文件转换成pb文件，以上操作请注意修改文件输入输出路径。也可以直接从obs桶中获取pb文件, 所有的文件均在：obs://randla-net/ACL/

2. pb转om

   ```
   1.上传pb文件到离线服务器上，根据pb的实际位置和名称修改run_pb2om中的pb位置和名称，以及设置om文件的输出位置：
   --model=/root/randlanet_final_version.pb --framework=3 --output=/root/randlanet --soc_version=Ascend310 --input_shape="xyz_0:3,40960,3;xyz_1:3,10240,3;xyz_2:3,2560,3;xyz_3:3,640,3;xyz_4:3,160,3;neigh_idx_0:3,40960,16;neigh_idx_1:3,10240,16;neigh_idx_2:3,2560,16;neigh_idx_3:3,640,16;neigh_idx_4:3,160,16;sub_idx_0:3,10240,16;sub_idx_1:3,2560,16;sub_idx_2:3,640,16;sub_idx_3:3,160,16;sub_idx_4:3,80,16;interp_idx_0:3,40960,1;interp_idx_1:3,10240,1;interp_idx_2:3,2560,1;interp_idx_3:3,640,1;interp_idx_4:3,160,1;rgb:3,40960,6" --log=info --out_nodes="probs:0" 
   2. bash ./run_pb2om.sh
   ```

3. 离线推理部署

   ```
   1. 参考此链接进行离线推理环境的部署即msame部署。
   https://gitee.com/ascend/modelzoo/wikis/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E5%B7%A5%E5%85%B7msame%E4%BD%BF%E7%94%A8%E6%A1%88%E4%BE%8B
   
   2. 若需要测试验证集的推理精度，则需要做如下数据准备：
      2.1 将输入数据以bin的形式进行存储，原模型对点云数据进行了随机采样操作，一次选取40960个点，再借助dataset的迭代器传入模型中进行测试或训练。注释掉训练和推理的代码，对数据进行初始化后，将1个step的数据进行保存。 
      也可以直接从obs桶中获取：obs://randla-net/ACL/bin_out_final/
      
      2.2 将存储好的bin格式文件上传至离线服务器上,以下列形式组织存放,总共有24个bin文件，推理是用到了其中21个，验证精度时用到了labels.bin文件
      ├── bin_out    
      |    ├──xyz_0.bin      
      |    ├──xyz_1.bin
      |    ├──xyz_2.bin
      |    ├──xyz_3.bin
      |    ├──xyz_4.bin      
      |    ├──neigh_idx_0.bin
      |    ├──neigh_idx_1.bin
      |    ├──neigh_idx_2.bin
      |    ├── ...
      
      2.3 再运行bash ./eval_inference.sh, 推理结果设置存放在bin_out中
      
      2.4 运行 python3 ./eval_acc_310.py ,即可得到验证集的推理精度。

   ```

3. 测试结束后会打印验证集的accuracy

   3.1 GPU上验证集精度结果如下：

   ```
    Initiating input pipelines
    Model restored from results/Log_2021-12-06_14-00-20/snapshots/snap-23501
    step0 acc:0.8991536458333333
    step1 acc:0.8917439778645834
    step2 acc:0.8771565755208334
    step3 acc:0.847216796875
    step4 acc:0.85198974609375
    step5 acc:0.82340087890625
    step6 acc:0.8893229166666666
    step7 acc:0.8707356770833333
    step8 acc:0.87530517578125
    step9 acc:0.8903971354166667
    ...
    step95 acc:0.8357218424479167
    step96 acc:0.8456258138020833
    step97 acc:0.88997802734375
    step98 acc:0.8664876302083333
    step99 acc:0.8704996744791667
   Confusion on sub clouds
   72.36 | 96.62 95.44 76.69 49.98 53.61 77.46 84.58 66.95 77.91 68.13 58.68 69.42 65.18 
   ```

   3.2 离线推理精度结果如下，与GPU上测试精度相近，离线推理精度达标：

   ```
    [INFO] model execute success
    Inference time: 1492.89ms
    [INFO] get max dynamic batch size success
    [INFO] output data success
    Inference average time: 1492.887000 ms
    [INFO] destroy model input success
    [INFO] unload model success, model Id is 1
    [INFO] Execute sample success
    [INFO] end to destroy stream
    [INFO] end to destroy context
    [INFO] end to reset device is 0
    [INFO] end to finalize acl

    acc:0.8622721354166667
    72.16 | 98.27 87.58 63.28 77.38 69.33 73.81 78.62 88.00 75.60 72.16 26.16 56.44 71.46
   ```
    3.3 可视化的代码也在eval_acc.py文件中，已经被注释，如需要可用来参考。

   


