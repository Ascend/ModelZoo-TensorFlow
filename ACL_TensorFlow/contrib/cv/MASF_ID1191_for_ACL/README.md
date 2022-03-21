# MASF_for_TensorFlow_ACL

## 离线推理

### 离线推理命令参考
./msame --model="/MASF/ngnn_acc.om" --input="/MASF/house/" --output="/MASF/out/" --outfmt BIN

### ckpt转pb程序参考
```
def main():
    ckpt_path = "/home/ma-user/modelarts/user-job-dir/code/itr152_model_acc0.5989558614143332"
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    tf.reset_default_graph()
    # 定义网络的输入节点
    model=MASF()
    input=tf.placeholder(tf.float32,shape=[None,227,227,3],name="input")
    # model.clip_value = FLAGS.gradients_clip_value
    # model.margin = FLAGS.margin
    model.KEEP_PROB = tf.placeholder(tf.float32)
    #
    model.weights = weights = model.construct_weights()
    #model.weights = weights = None
    #weights = get_weights()
    model.semantic_feature, outputs = model.forward_alexnet(input,weights)
    # 定义网络的输出节点
    acc = tf.identity(outputs, name='out')
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, '/home/ma-user/modelarts/user-job-dir/code/log/', 'model.pb')  # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            input_graph='/home/ma-user/modelarts/user-job-dir/code/log/model.pb',  # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names='out',  # 与定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='/home/ma-user/modelarts/user-job-dir/code/log/cartoon.pb',  # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')
    print("done")

```

### pb转om命令参考
atc --model=/MASF/cartoon.pb --framework=3 --output=/MASF/ngnn_acc --soc_version=Ascend310 --input_shape="input:1,227,227,3" --log=info --out_nodes="out:0"

### 图像转bin文件程序参考
```
import cv2
import os
dst_path='E:/PACS/binfile/giraffe'
x = 0
for root, dirs, files in os.walk('E:/PACS/kfold/cartoon/giraffe'):
    for d in dirs:
        print(d)  # 打印子资料夹的个数
    for file in files:
        print(file)
        # 讀入圖像
        img_path = root + '/' + file
        img = cv2.imread(img_path, 1)
        img=img.astype('float32')
        print(img_path)
        img.tofile(dst_path + "/" + file + ".bin")
```

### 推理结果

![精度结果](%E7%B2%BE%E5%BA%A6.png)

### 推理性能

![推理性能](%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD.png)

### 推理过程中的文件
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=+Ang3FM+ea8yFRyrtuC1YtByeWBzWWShkNrTxwShcutO2dDITmdx1NkPIPxZqu0xJ6T5N0GL2fg4sM027902p5r7VRmHB1s33N2eYIRDCwmcltigZbUhaZSm0sRPy0TQ1rPZLoyl97Ix2Mhou9VY6scMTUwcLJ8UsT4kZDPsj5MVX5DYrg5aH0kOwQ2JEqIbE2FlMKHCx4XTCnzqIQkBlN6Fw+mnoX4eW8iemFL9xOh8oprfocwJv+Zlzym5Jv2nskd+PH+SKDHDz/VV/3xBmO7aW5vgIkYUQjmuQBxGHpKi1LIBhFOCOSN2qVYXT09rQZTrePZYuVCjfwM/5/HmlLxb8S8UxHSLom1fUTOv7+H8TP3d/Q1coL+oOtx/0pxx1Watfso2ChLERiV9HytoiypUCwD5d79DtNH8JY7VTgPOKRCfHzWlb5yXit/lnQu+dup7+3/EIyfFk/l3LjiCk2NTaulvgISNcv/T0J+8rv2neQhAqXSSOy/eDeKc33i+wRfERUArLHWifVfYwvZFinzimyqwHcUnjSLjknuGtdg=

提取码:
123456

*有效期至: 2022/12/05 12:56:13 GMT+08:00