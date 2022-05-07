代码位置：https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow
参数：默认参数

训练命令：python training.py
GPU 训练时间：168小时
NPU 训练时间：24小时

GPU 精度：https://cann001.obs.cn-north-4.myhuaweicloud.com/log.1217215.out
NPU 精度：https://cann001.obs.cn-north-4.myhuaweicloud.com/modelarts-job-4c06ae91-0a9b-4c9c-a9f6-4e32fe5c0bd7-proc-rank-0-device-0.txt

ckpt转pb：https://gitee.com/ascend/modelzoo/blob/master/built-in/TensorFlow/Research/cv/image_segmentation/UNet3D_ID0057_for_TensorFlow/pbinference/unet3d_pb_frozen.py

pb转om：atc --model=/home/HwHiAiUser/AscendProjects/NGNN/pb_model/model.ckpt-163150.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/NGNN/ngnn_acc --soc_version=Ascend310 --input_shape="Placeholder:1,512,512,3" 

离线推理命令：./msame --model /home/HwHiAiUser/AscendProjects/tools/msame/ngnn_acc.om --input /home/HwHiAiUser/AscendProjects/tools/msame/982.bin --output /home/HwHiAiUser/ljj/AMEXEC/out/output1 --outfmt TXT --loop 2
离线推理结果：https://cann001.obs.cn-north-4.myhuaweicloud.com/ngnn_acc_output_0.txt
