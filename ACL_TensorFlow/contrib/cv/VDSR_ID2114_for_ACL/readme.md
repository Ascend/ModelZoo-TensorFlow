# 文件结构：
├── LICENSE                       //license  
├── modelzoo_level.txt            //进度说明文档  
├── README.md                     //说明文档  
├── caculate_zhibiao.py           //用于进行后处理，即用输出的bin文件（txt）计算精度  
├── ckpt2pb.py                    //用于1.ckpb转pb模型 2.pb转pbtxt 3.pbtxt转pb  
├── codename.py                   //用于输出模型的所有节点的名字  
├── generate_bin.py               //用于把输入数据转换成bin文件  
├── pb_test.py                    //用于在pb模型上测试选的输入输出节点好不好使，得出精度证明可以  


# 模型功能：
对放大后的模糊图像进行处理以变清晰

# 推理过程与说明
1.测试集转bin  
运行generate_bin.py，把四个测试集的mat图片均转为bin形式  
附录1提供所有测试集mat图片及所有转好的bin文件  
2.ckpt转pb  
运行ckpt2pb.py的50到55行，把ckpt模型转成pb格式的模型  
附录2提供一个ckpt模型  
3.pb模型的处理    
运行ckpt2pb.py的57到62行，把pb模型转为pbtxt  
对pbtxt进行修改，改为自己想要的模型结构  
运行ckpt2pb.py的64到70行，pbtxt转回pb    
附录3提供一个调整后的pbtxt文件及pb模型（去掉正常模型结构中的数据预处理相关部分，直接把输入节点定为神经网络的入口（fifo_queue_DequeueMany 节点改为Placeholder），且把输入形状定为276x276，其他形状的图片不可用该模型测试！！！）  
4.pb转om  
参考命令如下：  
atc --model=/home/VDSR/om/model.pb --framework=3 --output=new0 --soc_version=Ascend310 --out_nodes="shared_model/Add"  
附录4提供由上一步得到的pb模型转换好的om模型  
5.msame推理  
参考命令如下：  
./msame --model /home/VDSR/om/new0.om --input /home/VDSR/inbin/3_2.bin --output /home/VDSR/outbin --outfmt TXT --loop 2  
附录5提供3.2bin（由set5测试集的3_2.mat转换而来,它的形状是276x276）及它的推理结果new0_output_0.txt  
6.验证  
运行caculate_zhibiao.py，计算精度  
计算精度需要输入的是原图、原图放大后的图及模型处理后的放大图  
附录6提供set5数据集中3号原图3.mat、放大2倍后的图3_2.mat、3_2.mat推理结果new0_output_0.txt  
用他们得到的推理精（前为该图片的gpu实现精度，后为离线推理精度）、性能如下图所示：  
![输入图片说明](GPUjingdu.png)  
![输入图片说明](tuilijingdu.png)  
![输入图片说明](tuilixingneng.png)  


# 其他说明  
1.以上提供的代码均是在modelart云上跑的，多了些与obs桶传输相关的语句  
2.测试其他图片的精度时需要更改pb模型输入节点的形状、更改caculate_zhibiao.py的71到83行（代码里有说明）、变更对应的输入图片  
3.所有测试图片的尺寸说明、及他们的gpu精度见附录7  
# 附录  
URL:  
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=zDgBLssF084S5Ou6pafVbFaeLYg2yRDHtZBYcRcjv3R45FKtQ5P31bZtHJVseMMdTtO5MKB9uzD1MJ7BA5m5bYPqEEFeH68Vax5Mvk/7K1mJL5vqF+5xWBFsKIb02cZW3s1kk63sLOf1n+mRY0pPxKxNm+iaOaBKVvCEd/HW1BlYItCHApuF9vIc+34FMTl4SMTHWkDVnYkZOIY6d6Uk+zwZ2B/j3+hpjdGQifCwl3TmkwwUoj3Z8JTRLGQZQLep9xQb1LM9p3jb4UmrPMMTPVU3rjMA9OYMuejnoN0Dq03blKMzQV4ea2LeTMtjSAovdJLvqWkPY8VG9VjaxAA/rjU2grROx31SJAUySsiwwp1XI0F9fnaXhvLXatD2vs2u8EDAb9tteCEVqQVI/OQJtFEOTCstugvZ61C+Pt4zRed/g4HfxZUfhdfAGRCRaRILvdCRo5A3ukEpI9bu4S61p2DYWWpPQ76+r4QXaYLe9xWyNApVSYSC28zlV0h6ZleB5/2CkI5VYNLoyJufm4LW8qUjq17/JZMBAir1VoOXC4ukVKT68IkSH+Y4vKraGhHjbYOWS1xnhJkpNmuD5FSoUg== 
 
提取码:111111  
有效期至: 2022/12/06 15:49:49 GMT+08:00