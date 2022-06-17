## 推理/验证过程<a name="section1465595372416"></a>

1. ckpt文件转pb文件（修改训练文件中src/squeezeseg_pb_frozen.py文件）

   ```
   #将此处的ckpt路径改成需要转换的ckpt文件路径
   tf.app.flags.DEFINE_string('checkpoint_path', "./inference/model.ckpt-27000",
                        """Directory of checkpoint. """)
                        
   也可以直接从obs桶中获取pb文件frozen_pb，为训练25500steps所得ckpt文件转换而得：obs://squeezeseg-training/SqueezeSeg/bin_out
   ```

2. pb转om

   ```
   1.上传pb文件到离线服务器上，根据pb的实际位置和名称修改run_pb2om中的pb位置和名称，以及设置om文件的输出位置：
   --model=/home/HwHiAiUser/AscendProjects/frozen_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/squeezeseg_acc 
   2. bash ./run_pb2om.sh
   ```

3. 离线推理部署

   ```
   1. 参考此链接进行离线推理环境的部署即msame部署。
   https://gitee.com/ascend/modelzoo/wikis/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E5%B7%A5%E5%85%B7msame%E4%BD%BF%E7%94%A8%E6%A1%88%E4%BE%8B
   
   2. 若需要测试验证集的推理精度，则需要做如下数据准备：
      2.1 将输入数据input,mask以及label以bin的形式进行存储()
        lidar_per_batch.tofile("xxx/bin_out/input/{0:05d}.bin".format(i))
    	lidar_mask_per_batch.tofile("xxx/bin_out/input_mask/{0:05d}.bin".format(i))
        label_per_batch.tofile("xxx/bin_out/input_label/{0:05d}.bin".format(i))
        
      也可以直接从obs桶中获取：obs://squeezeseg-training/SqueezeSeg/bin_out
      
      2.2 将存储好的bin格式文件上传至离线服务器上,以下列形式组织存放
      ├── bin_out    
      |    ├──lidar      
      |    ├──lidar_mask
      |    ├──label
      |    ├──pred_cls //为新建的空文件夹，用于存储下一步验证集的推理结果。
      
      2.3 再运行bash ./eval_inference.sh, 推理结果设置存放在bin_out/pred_cls中
      
      2.4 运行 python3 ./eval_acc_310.py ,即可得到验证集的推理精度。
      
   3.若无需测试验证集的推理精度，则无需执行步2，直接执行该步：
   
      3.1 新建文件夹demo_data以及子文件夹demo_npy
      ├── demo_data    
      |    ├──demo_npy
      
      3.2 直接上传.npy文件至demo_npy中，再修改demo.py中相应的路径。
      
      3.3 修改demo.py,detect()函数的传参设置为False,即detect(False)
      
      3.4 运行demo.py文件 python3 ./demo.py
   ```

4. 测试结束后会打印验证集的accuracy，取一个训练25000step左右的ckpt进行测试。

   4.1 NPU上验证集精度结果如下：

   ```
   INFO:tensorflow:Restoring parameters from /cache/npu_ckpt_log/model.ckpt-25500
   INFO:tensorflow:Restoring parameters from /cache/npu_ckpt_log/model.ckpt-25500
   Evaluation summary:
     Timing:
       read: 0.002s detect: 0.356s
     Accuracy:
       car:
   	Pixel-seg: P: 0.660, R: 0.966, IoU: 0.645
       pedestrian:
   	Pixel-seg: P: 0.363, R: 0.268, IoU: 0.182
       cyclist:
   	Pixel-seg: P: 0.266, R: 0.710, IoU: 0.240
   ```

   4.2 离线推理精度结果如下，与NPU上测试精度相同，离线推理精度达标：

   ```
   Inference time: 677.725ms
   [INFO] get max dynamic batch size success
   [INFO] output data success
   [INFO] destroy model input success
   [INFO] start to process file:./bin_out/input/02790.bin
   [INFO] start to process file:./bin_out/input_mask/02790.bin
   [INFO] model execute success
   Inference time: 677.713ms
   [INFO] get max dynamic batch size success
   [INFO] output data success
   [INFO] destroy model input success
   Inference average time : 677.67 ms
   Inference average time without first time: 677.63 ms
   [INFO] unload model success, model Id is 1
   [INFO] Execute sample success
   [INFO] end to destroy stream
   [INFO] end to destroy context
   [INFO] end to reset device is 0
   [INFO] end to finalize acl
   [0.99729069 0.6595094  0.36308151 0.26606843]
       car:
           Pixel-seg: P: 0.660, R: 0.966, IoU: 0.645
       pedestrian:
           Pixel-seg: P: 0.363, R: 0.268, IoU: 0.182
       cyclist:
           Pixel-seg: P: 0.266, R: 0.709, IoU: 0.240
   ```

   


