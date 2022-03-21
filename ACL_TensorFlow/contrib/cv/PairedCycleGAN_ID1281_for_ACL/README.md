# PairedCycleGAN_for_TensorFlow_ACL

## 离线推理
### 所使用数据集
数据集为网络上寻找的MA人像面部妆容数据集

数据集获取链接为：[obs分享链接](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=oW2yZMl66FH/ghTU6+mh7RENEwQ1NxF7PFw/Hf9SHP8SR8C3s86LxYBaWwRN+h9pyT+FFiRWHwXGqgXU0mHW+JxW2y+L6d9z2AFzvCQnpYoJB8Rkr6dqSU/LjjQ/NnKH/toWHRIy8HndxNfxkSMPO8kbwGoEfQoCRi9wNPfTmb3Hp4auBaDWJr+CAgziVlxJJvWxHpuatLSGoMVPQkt0ZG5ZLEEEtb8H2L/pKExoS79PC92mItqidzrLsX5i859wZVsl5EaRf1SxlIqdZNc/FtbWI4fkxVfMSdG/35xPsFOC7kgQLtGvFrhjGFBjWTMbEku6DlDUEMlNvkKjn1jNxx52B/SUG1Jm6vKTVuqlXUtWW6bnEK0f+17dVLZBBvmjmnchJfCVxXEb5+NcQgnTfEwglxgFOS0hcYL6UhN4FRrbQEbsJb0cb4u/XSMJ5eC3roLRLehZ4hCq+7di/qJUbsRt8RAnIkVZvy+yai9muD9oJEZ4E5jZKAgbhE8zZ5mCNxRHbl1GlZkgoCYNQ0rGhVvxSk51dfhC2n8z3KRwj04zhsWm5PyDtZHbfsqy2xAFcOy2Iv5IJYbDaKho7sgiREmJqwzlaLtns9HvbvrjggjNFHEaC/0Uow2L4pd8hur2G4+T7SXyAdEiIWLqKzmCnw==)
提取码:
123456

*有效期至: 2023/02/10 22:21:06 GMT+08:00
### 离线推理命令参考
./msame --model "/root/PairedCycleGAN/project/test0113_1/om/PairedCycleGAN.om" --input "/root/PairedCycleGAN/project/test0113_1/image/out/makeup/000000.bin,/root/PairedCycleGAN/project/test0113_1/image/out/non-makeup/000000.bin" --output "/root/PairedCycleGAN/project/test0113_1/image/out/" --outfmt BIN  --loop 1
### ckpt模型
在910A处理器上训练90个Epoch得到的ckpt模型文件
模型文件链接URL:[obs分享链接](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=oW2yZMl66FH/ghTU6+mh7RENEwQ1NxF7PFw/Hf9SHP8SR8C3s86LxYBaWwRN+h9pyT+FFiRWHwXGqgXU0mHW+JxW2y+L6d9z2AFzvCQnpYoJB8Rkr6dqSU/LjjQ/NnKH/toWHRIy8HndxNfxkSMPO8kbwGoEfQoCRi9wNPfTmb3Hp4auBaDWJr+CAgziVlxJJvWxHpuatLSGoMVPQkt0ZG5ZLEEEtb8H2L/pKExoS79SvA/MZuSHfMb1n5s5GwSU+WrskFKfwq8e6iwmogj+XT2h6jb+5u9eU/KiA2toJ1ekPlkqYS3TYuck730cZ9Gdmy2K9TLfjohOy9qg8agdZpNXIxUCeSQWX+KlmrjtjwT4k+PgJe9O0q1eWgpH1myD0c4Iw0nbwmMFtXKovNuDAIbDo35jS1WiPOEd12cw8DyL975BK5HbhvmaFjk23fT4M5YoE3OeNoBGaL6PKK9ijRZ04A6fKeq4qMjXSicvLQTS9jIhtOnr2wvSLvsZgXz3+8p3FSmP+nUw254UliWzxtOqtcJDqK4vgWH4KDKDuA2zNwp2qmUwjQfn2ixHYBDHBfrby0ljoESI17Khd7gTAHwYo1hYDU44+HYwMNLFBZHhmVE8mrI5jo0GFv7qoDONe1vKBryCBvAA6RwjSl0VDX5pLyV61L0TUgMq76aby8tipSTBe1OxFjraYGHyzpqFR4XMK4CqmYTD/EAoZ/HdxA==)

提取码:123456

*有效期至: 2023/02/10 22:05:54 GMT+08:00
### pb模型
将ckpt模型传入模型转换程序，得到pb模型。

pb模型文件链接：[obs分享链接](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=oW2yZMl66FH/ghTU6+mh7RENEwQ1NxF7PFw/Hf9SHP8SR8C3s86LxYBaWwRN+h9pyT+FFiRWHwXGqgXU0mHW+JxW2y+L6d9z2AFzvCQnpYoJB8Rkr6dqSU/LjjQ/NnKH/toWHRIy8HndxNfxkSMPO8kbwGoEfQoCRi9wNPfTmb3Hp4auBaDWJr+CAgziVlxJJvWxHpuatLSGoMVPQkt0ZG5ZLEEEtb8H2L/pKExoS78aB7QP366zOU6/aGHUxtAv3LvVZ/3p2zQaKLAt9uG1tCqSyGQvo6XoVUm9zpNe9Kj5shxmSNqr5XJwXdDsPczF2YqmDhokW8ROtGvWklsj/w0eS5kL+3SZmu7N40ptGVARsUOubtfiTFSpEPhdGkZDMnblNvXoGP50sHreb4OclfJ5RjtAYlrRDHze6dG/foo+6qCgSzHSoV6FeKGDCXm4LUREVhsGNmgp3ngiNl+InthSAg3J0a1mVvh5uSHy+Hj/IksWljRl6k7fzIEjOPSSH+np3UVQDLTQJcRelj26X5Q6q73z4HA8gscNnebeR/1meKQwK4kOoT+FL3qLyspFzcnS/EJB/qufqn6w7bDxkC6h8L0UMz6MPh3xCRB5gR3tTRqy+nahsgg6hP1yX9j2rl7m9rUK1B8jg/Oc0xUY51ILTk8ZPcgxa2b18vssd98=)
提取码:123456

*有效期至: 2023/02/10 22:08:02 GMT+08:00
### ckpt转pb程序参考
```
def freeze_graph(input_checkpoint ,output_graph):
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    # output_node_names = "Model/generator/r9/c2/conv2d/kernel/Adam_1"
    output_node_names = "Model/generator/B_t"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def  ,# 等于:sess.graph_def
            output_node_names=output_node_names.split(",")  )# 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

# input_checkpoint='inceptionv1/model.ckpt-0'
# out_pb_path='inceptionv1/frozen_model.pb'

input_checkpoint ='C:/Users/xkh/Desktop/beautyGAN-100'
out_pb_path ='./results/frozen_model.pb'
freeze_graph(input_checkpoint, out_pb_path)

```
### om模型
将pb模型上传到云开发环境，利用ATC模型转换工具进行模型转换

om模型分享链接：[obs分享链接](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=oW2yZMl66FH/ghTU6+mh7RENEwQ1NxF7PFw/Hf9SHP8SR8C3s86LxYBaWwRN+h9pyT+FFiRWHwXGqgXU0mHW+JxW2y+L6d9z2AFzvCQnpYoJB8Rkr6dqSU/LjjQ/NnKH/toWHRIy8HndxNfxkSMPO8kbwGoEfQoCRi9wNPfTmb3Hp4auBaDWJr+CAgziVlxJJvWxHpuatLSGoMVPQkt0ZG5ZLEEEtb8H2L/pKExoS7++NPb06W9fogJK8XF93j3fVsXsAu6FRsNxMcbNJaagBjGU+pcNfxwUoqnxls4SEHq0mF5aGSH6TZdUfhirUjnmJfgH8zG3kcUqv86I3e3tgnwUpMSEsLiPXEer8dNAbjTjtboJDY1L46ZfS3OULb1/J5mx/yuUV1BfeGVlVsORrv739YBCKL/gMb1avboVZH2Xsy4027oiCSHBlHzmD+6uYNiqj6D1fvOJvkPALdVlgIqMsov8dzDk45YPYoHbtfh8I7EarqleZzR9moVv03iu9mvm+5XG5hkJKK+3gVsbzX8p+IP8js9BEIpLsHZeFABa/GRnWXEC6ereI3U62LNSdMyvoEngmqvoCnt6JSk/08tXkGz9dbXRsVE5zYEa43WCBIjG5UkoOcSh0lvbK9AFx4MTivQ+dPU+72NjTFSDIotELTxjNjuNmKPpLC1xMCo=)
提取码:123456

*有效期至: 2023/02/10 22:10:41 GMT+08:00
### pb转om命令参考
atc --model=/root/PairedCycleGAN/project/test0113_1/pb/frozen_model.pb --framework=3 --output=/root/PairedCycleGAN/project/test0113_1/om/PairedCycleGAN --soc_version=Ascend310 --log=info --input_shape="input_A:1,256,256,3;input_B:1,256,256,3"
### 图像转bin文件命令参考
```
python3 img2bin.py -i /root/PairedCycleGAN/project/test0113_1/image/makeup/ -w 256 -h 256 -f RGB -a NHWC -t float32 -m [0,0,0] -c [1,1,1] -o /root/PairedCycleGAN/project/test0113_1/image/out/makeup

python3 img2bin.py -i /root/PairedCycleGAN/project/test0113_1/image/non-makeup/ -w 256 -h 256 -f RGB -a NHWC -t float32 -m [0,0,0] -c [1,1,1] -o /root/PairedCycleGAN/project/test0113_1/image/out/non-makeup
```
## 使用msame工具进行推理
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

- Ascend310 离线推理
NPU离线推理单个图片耗时约983ms 
![输入图片说明](../../../../../%E5%9B%BE%E7%89%87.png)