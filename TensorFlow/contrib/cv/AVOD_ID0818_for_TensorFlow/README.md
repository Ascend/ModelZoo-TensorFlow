Delivery Result for GPU reproduction: [README_gpu](1_gpu_training/README_gpu.md) and support documents in folder [1_gpu_training](1_gpu_training)

Delivery Result for NPU training (currently, conversion, loss convergence, accuracy): 

Support documents in folder [2_npu_training](2_npu_training)
The code in this repo is the current working code for NPU training.

## Training Dataset
The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```

Pre-process data by following the [README](README.md)

To Train on other datasets, either write a diff dataloader or convert to the kitti format.

## Evaluation AP
### KITTI Object Detection Results (3D and BEV) Car
|               |            |          |   |           |        AP-3D  |           |   |           |       AP-BEV  |           |
|:-------------:|:----------:|:--------:|---|:---------:|:-------------:|:---------:|---|:---------:|:-------------:|:---------:|
|               | **Runtime**|**Steps** |   |  **Easy** | **Moderate**  |  **Hard** |   |  **Easy** | **Moderate**  |  **Hard** |
|     Paper     |    0.10    |          |   |**81.94**  |    **71.88**  | **66.38** |   |   88.53   |      83.79    |   77.90   |
|      GPU      |    0.9     | 120,000  |   |   80.97   |     67.27     |   65.61   |   | **89.56** |   **86.33**   | **79.60** |
| NPU <br>`no mix-percision`|    3.5     | 120,000  |   |   77.61   |     67.40     |   65.74   |   |   88.74   |   85.44       |   78.96   |
| NPU <br>`mix-percision`   |    1.0     | 200,000  |   |   76.35   |     67.18     |   66.49   |   |   88.74   |   85.44       |   78.96   |
|200dk batch=1; including pre&post process |0.653|||76.58|    67.25     |   66.36   |   |   88.68   |   85.45       |   78.84   |

### Summary
Overall performance is worse than GPU, it's a trade-off between speed and accuracy. To meet the same accuracy with mix-percision, it's required to train extra steps; otherwise without mix-percision, the runtime for each step is much slower.

### Code changes after using Conversion Tool:  
| Issue | Code change|
|-------|------------|
| path_drop_probabilities | initially set to 1.0;  |
|tf.contrib.memory_stats.MaxBytesInUse() not supported | remove |
|missing npu config|custom_op.name = "NpuOptimizer";rewrite_options.remapping; rewrite_options.memory_optimization; |
|Error Caused by: Pad BEV input from 700 to 704 to allow even divisions for max pooling; Pad + conv2d -> somehow pad operation seems to be fused into conv2d, causing shape issue when backpropgation| put padding operation outside of model |
|Error Caused by: resize input image in model | move out to pre-processing & set input to static |
| Dynamic shape caused by `bool_mask` - `mb_mask` | regularize the mask to static shape `[1024]`   |
| Dynamic shape in (`anchors_info`) and (`label_anchors`, `label_boxes_3d`, `label_classes`)| Padding anchor to a max static shape `30000`, `20`|
|Tf.case tf.cond seems also not working well in backprob|move the condition outside of the model|
| `mixed_precision` can only be used after the model weights saved then load once (the first time won't work) `[ERROR] RUNTIME(8532)kernel task happen error, retCode=0x26, [aicore exception].` | use with at least one checkpoint|
| `profiling_options string` cannot have revered `'` and `"` e.g. must be `'{"output":"path","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"img_input/sub","bp_point":"train_op/gradients/bev_vgg_pyr/conv1/conv1_1/Conv2D_grad/Conv2DBackpropFilter"}'` | cannot be `"{'output':'path',}"` |
Analysis: all the issues are somehow related to dynamic shape or if-condition, not likely to be resolved by the code conversion tool

## Inference
### Note
For model conversion, one line (https://github.com/Ascend-Huawei/AVOD/blob/main/avod/core/models/avod_model.py#L513) needs to be commented out because it generates dynamic output which causes segementation fault error.
Then `cd pb_model; bash freeze.sh`

### Download Models and Validation Set
Put pb and om into `3_inference/code/model`

Put Val set into `3_inference/code/data` like <br>

### Conversion Command
```
cd 3_inference/code/

atc --input_shape="bev_input/bev_input_pl:704,800,6;img_input/img_input_pl:360,1200,3;pl_anchors/anchors_pl:89600,6;pl_anchors/bev_anchor_projections/bev_anchors_norm_pl:89600,4;pl_anchors/img_anchor_projections/img_anchors_norm_pl:89600,4;pl_anchors/sample_info/frame_calib_p2:3,4;pl_anchors/sample_info/ground_plane:4" --input_format=NHWC --output model/avod_npu --soc_version=Ascend310 --framework=3 --model model/avod_npu.pb
```
### Run Inference
Download Validation Set, then

```
cd 3_inference/code/src/ 
python main.py
```

### Run Kitti Eval
```
cd 3_inference/code/src/ 
python kitti_eval.py
```
