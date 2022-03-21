## 3d-pose-baseline

### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：**Human Pose Estimation

**版本（Version）：1.0**

**修改时间（Modified） ：2022.3.6**

**大小（Size）：16.7M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的3d-pose-baseline姿态估计网络离线推理代码**

### 概述

3d-pose-baseline是一个经典的2d-to-3d人体姿态预测网络，同时也是3d人体姿态预测的一个重要baseline。该模型的体系结构借鉴了许多这些年来深度神经网络优化方面的改进，包括但不限于(1)使用2d/3d点作为输入输出，而不是原始图像、2d概率分布作为输入，3d概率、3d动作信息、姿态系数作为输出，这能显著降低模型的收敛难度和训练时长；(2)根据模型的特点而采用已经被广泛使用的Leaky-Relu激活函数、残差连接和最大归一约束等模型参数或构造，以取得最优的模型性能。3d-pose-baseline证明了仅需要一个很简单的模型架构，就能从人体2d骨骼点中还原出其在3d空间中的骨骼点坐标。

- 参考论文：

  [Martinez, Julieta et al. “A Simple Yet Effective Baseline for 3d Human Pose Estimation.” *2017 IEEE International Conference on Computer Vision (ICCV)* (2017): 2659-2668.](https://arxiv.org/pdf/1705.03098.pdf)

- 参考实现：

  [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}		# 克隆仓库的代码
  cd {repository_name}    		# 切换到模型的代码仓目录
  git checkout {branch}			# 切换到对应分支
  git reset --hard {commit_id}	# 代码设置到对应的commit_id
  cd {code_path}					# 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

### 文件结构
  ```
  |- src                                     # 包含训练、ckpt转pb等前期代码
  |- 3d_pose_baseline.om                     # 由pb模型转换而来的om模型
  |- encoder_inputs.bin                      # 原始数据集转换而来的bin文件
  |- encoder_inputs_1batch.bin               # 只从整个数据集中抽取了1个batch的bin文件
  |- frozen_model.pb                         # 冻结了参数和模型结构后的pb模型
  |- model.pb                                # 由ckpt转换而来的pb模型
  |- ......
  ```

### ckpt转pb
在本机运行时，我们设置了`--freeze_pb`参数来启动NPU推理阶段的代码，但我们强烈推荐使用ModelArts的运行方式，此时我们可以简单的使用`src/scripts/freeze_pb.sh`来启动代码。

此时，请使用`src/predict_3dpose.py`中的如下代码：
```
# === save the model graph to .pb file ===
model.saver.save(sess, os.path.join(train_dir, 'checkpoint'))
tf.train.write_graph(sess.graph_def, os.path.join(train_dir, 'pb_model'), 'model.pb', False)

if ckpt and ckpt.model_checkpoint_path:
   # Check if the specific checkpoint exists
   if FLAGS.load > 0:
      if os.path.isfile(os.path.join(ckpt_path, "checkpoint-{0}.index".format(FLAGS.load))):
         ckpt_name = os.path.join(ckpt_path, "checkpoint-{0}".format(FLAGS.load))
      else:
         raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
   else:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
   print("use the ckpt file:{0}".format(ckpt_name))
   freeze_graph.freeze_graph(
      input_graph=os.path.join(train_dir, 'pb_model', 'model.pb'),
      input_saver='',
      input_binary=True,
      input_checkpoint=ckpt.model_checkpoint_path,
      output_node_names='linear_model/add_1',
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=os.path.join(train_dir, 'pb_model', 'frozen_model.pb'),
      clear_devices=True,
      initializer_nodes=''
      )
else:
   print("Could not find checkpoint. Aborting.")
   raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))
```

注意，此时我们应该确保模型图中没有无谓的输入，因此将`src/linear_model.py`中的模型定义略加修改。
```
# self.isTraining = tf.compat.v1.placeholder(tf.bool, name="isTrainingflag")
self.isTraining = False
# self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
self.dropout_keep_prob = 1.0
```
我们提供转换好的[pb模型文件](https://pan.baidu.com/s/1AFCo5XTn4GifEJfMaFKqPQ?pwd=0795)。

### pb转om
使用ATC模型转换工具进行模型转换时可以参考如下指令:
```
atc --model=/home/HwHiAiUser/AscendProjects/pb_model/frozen_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/om_model/3d_pose_baseline --soc_version=Ascend310 --input_shape="inputs/enc_in:128,32" --log=error --out_nodes="linear_model/add_1:0"
```

我们提供转换好的[om模型文件](https://pan.baidu.com/s/1OGO7TYYSg2UoAIjPRBpcqw?pwd=0795)。

### msame工具
我们采用msame工具进行离线推理，参考[msame简介](https://gitee.com/ascend/tools/tree/master/msame), 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

### 数据集转bin
该过程对应`src/predict_3dpose.py`中的如下代码：
```
# === Load training batches for one epoch ===
encoder_inputs, decoder_outputs = model.get_all_batches(train_set_2d, train_set_3d, FLAGS.camera_frame, training=False)
encoder_inputs = np.array(encoder_inputs)
encoder_inputs_np = encoder_inputs.reshape((-1, 32))
encoder_inputs.tofile(os.path.join(train_dir, "encoder_inputs.bin"))
encoder_inputs_1batch = encoder_inputs[0]
encoder_inputs_1batch.tofile(os.path.join(train_dir, "encoder_inputs_1batch.bin"))
```
也就是说，我们在获取pb模型的同时，也获得了转换好的bin格式数据集。

为了不同程度的测试需求，我们提供完整数据集和1 batch两种大小的bin文件,这是我们转换好的[bin文件](https://pan.baidu.com/s/1XZHDUvW1bZKPxXDhmtMqlQ?pwd=0795)。

### 推理测试
使用msame推理工具，参考如下命令，发起推理测试：
```
./msame --model "/home/HwHiAiUser/AscendProjects/om_model/3d_pose_baseline.om" --input "/home/HwHiAiUser/AscendProjects/bin_data/encoder_inputs_1batch.bin" --output "/home/HwHiAiUser/AscendProjects/out/" --outfmt TXT
```
然后，我们需要将获取到的推理结果，经过如下代码去归一化：
```
enc_in[bidx] = data_utils.unNormalizeData(enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d)
dec_out[bidx] = data_utils.unNormalizeData(dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d)
poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
```
并通过如下代码将其转换到世界坐标系中：
```
# Convert back to world coordinates
if FLAGS.camera_frame:
   N_CAMERAS = 4
   N_JOINTS_H36M = 32

   # Add global position back
   dec_out = dec_out + np.tile(test_root_positions[key3d], [1, N_JOINTS_H36M])

   # Load the appropriate camera
   subj, _, sname = key3d

   cname = sname.split('.')[1]  # <-- camera name
   scams = {(subj, c + 1): rcams[(subj, c + 1)] for c in range(N_CAMERAS)}  # cams of this subject
   scam_idx = [scams[(subj, c + 1)][-1] for c in range(N_CAMERAS)].index(cname)  # index of camera used
   the_cam = scams[(subj, scam_idx + 1)]  # <-- the camera used
   R, T, f, c, k, p, name = the_cam
   assert name == cname

   def cam2world_centered(data_3d_camframe):
   data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
   data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M * 3))
   # subtract root translation
   return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, N_JOINTS_H36M))

   # Apply inverse rotation and translation
   dec_out = cam2world_centered(dec_out)
   poses3d = cam2world_centered(poses3d)
```
最后，可视化后可以得到类似如下结果：
![可视化结果](src/imgs/viz_example.png)

### 推理精度
我们经过多次实验求取平均值后，与原论文的精度对比数据如下表所示，其评价指标为MSE，单位为mm。
| 原论文 | NPU离线推理 |
| :----: | :--------: |
|  47.7  |    48.1    |
   
   
