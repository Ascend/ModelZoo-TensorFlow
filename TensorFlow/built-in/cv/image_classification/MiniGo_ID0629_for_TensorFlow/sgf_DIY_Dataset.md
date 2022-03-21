# MiniGo：自制数据集

### Step1：获取sgf围棋棋谱

自行搜寻  / 获取 sgf 围棋棋谱文件。



### Step2：修改dual_net.py

**1.将原 dual_net.py 脚本中第272-279行代码：**

```
def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)
```

修改为：

```
def get_features_planes():
    return features_lib.AGZ_FEATURES_PLANES
```

**2.将原 dual_net.py 脚本中第282-289行代码：**

```
def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)
```

修改为：

```
def get_features():
    return features_lib.AGZ_FEATURES
```



### Step3：检查 sgf 文件内容

实验发现若 sgf 文件中存在中文会导致数据集制作失败，因此此步骤检查 sgf 文件内容并将中文字符替换为空字符。

运行sgf_file_check.py

```
python3 sgf_file_check.py --sgf_path /sgf/path
```



### Step4：制作数据集

使用step3中检查后的 sgf 文件，运行 sgf_to_tfrecord.py 即可将 sgf 文件转换为 tfrecord 文件，自制数据集。

```
python3 sgf_to_tfrecord.py --sgf_path /sgf/path
```

