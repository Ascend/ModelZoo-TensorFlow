GPU代码位置：https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow


NPU代码：

Files

├── __data__ here the data should be stored  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   └── __tusimple_dataset_processing.py__ processes the TuSimple dataset  
├── __doc__ documentation  
├── __inference_test__ inference related data  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   └── __images__ for testing the inference    
├── __trained_model__  pretrained model for finetuning  
├── __clustering.py__ mean-shift clustering  
├── __datagenerator.py__ feeds data for training and evaluation  
├── __enet.py__ [Enet architecture](https://github.com/kwotsin/TensorFlow-ENet)  
├── __inference.py__ tests inference on images  
├── __loss.py__ defines discriminative loss function  
├── __README.md__  
├── __training.py__ contains training pipeline  
├── __utils.py__ contains utilities files for building and initializing the graph  
└── __visualization.py__ contains visualization of the clustering and pixel embeddings  


参数：默认参数

数据集：http://benchmark.tusimple.ai/#/t/1

 **训练命令：npu_train.sh** 

GPU 训练时间：168小时
NPU 训练时间：24小时

GPU 精度：https://cann001.obs.cn-north-4.myhuaweicloud.com/log.1217215.out

step_valid 18150 valid loss 0.09118794 	valid l_var 0.07415567 	valid l_dist 0.0039632297 	valid l_reg 0.013069032

NPU 精度：https://cann001.obs.cn-north-4.myhuaweicloud.com/modelarts-job-4c06ae91-0a9b-4c9c-a9f6-4e32fe5c0bd7-proc-rank-0-device-0.txt

step_valid 18150 valid loss 0.074521884 	valid l_var 0.057265442 	valid l_dist 0.0004725394 	valid l_reg 0.016783904

ckpt转pb：https://gitee.com/ascend/modelzoo/blob/master/built-in/TensorFlow/Research/cv/image_segmentation/UNet3D_ID0057_for_TensorFlow/pbinference/unet3d_pb_frozen.py

pb转om：atc --model=/home/HwHiAiUser/AscendProjects/NGNN/pb_model/model.ckpt-163150.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/NGNN/ngnn_acc --soc_version=Ascend310 --input_shape="Placeholder:1,512,512,3" 

离线推理命令：./msame --model /home/HwHiAiUser/AscendProjects/tools/msame/ngnn_acc.om --input /home/HwHiAiUser/AscendProjects/tools/msame/982.bin --output /home/HwHiAiUser/ljj/AMEXEC/out/output1 --outfmt TXT --loop 2

离线推理结果：https://cann001.obs.cn-north-4.myhuaweicloud.com/ngnn_acc_output_0.txt
