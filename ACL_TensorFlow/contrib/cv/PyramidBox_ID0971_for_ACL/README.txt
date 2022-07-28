
原始pb模型：
https://cann-id0971.obs.cn-north-4.myhuaweicloud.com:443/Inference/creat_input.pb?AccessKeyId=H0UI7M1LM37WKJ6NUOJY&Expires=1674551687&Signature=33leEGhuUpXnHReQARvaC4H0dAc%3D

ATC转换命令：
atc --model=/home/HwHiAiUser/AscendProjects/PyramidBox/creat_input.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/PyramidBox/out_290000 --soc_version=Ascend310 --input_shape="pyramid_box_preprocessing_eval/resize/ResizeBilinear:1,640,640,3"

转换的om模型：
https://cann-id0971.obs.cn-north-4.myhuaweicloud.com:443/Inference/out_290000.om?AccessKeyId=H0UI7M1LM37WKJ6NUOJY&Expires=1689421431&Signature=Fh8SB7DOAJkNMIZxwrjcBGurzgw%3D

