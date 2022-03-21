Deliverable results for GPU training: [README_GPU](https://rnd-gitlab-ca-g.huawei.com/hispark/model_training_hq/-/blob/master/Lip2Wav/1_gpu_training/README_GPU.md), supporting document in folder [1_gpu_training](1_gpu_training)

**[Google Drive link for checkpoint files and .pb files](https://drive.google.com/drive/folders/13dnqFc3WtEFE9dCbvVNQd4q5sDsbFmmF?usp=sharing)**

Deliverable results for NPU training and supporting document in folder [2_npu_training](2_npu_training)

## NPU Training :

### Dataset Download:

To download a speaker dataset, run sh download_speaker.sh Dataset/<speaker> (replace  with one of five speaker option). Then run preprocess.py to preprocess the raw dataset. We provide a preprocessed chess dataset from our experiment, it is stored on the GPU / NPU servers. Below is the location for the NPU preprocessed chess dataset:

 - Ascend NPU server (jiayansuo): 

    original: `/data/lip2wav/chess`

    softlnk: `/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/Dataset/chess`

  
### NPU training command
  
```
cd scripts
bash run_npu_1p.sh
```

### Code changes after using conversion tool:

| Observed Issues  | Code Changes | 
| --------  | ------------------- |
| *Datafeeder (feeder.py):* <br/> NPU does not support tf.FIFOQueue for data buffering and queueing. | Changed to buitlin python queue implementation class::Queue for building training and evaluation queues. During training (in synthesizer/train.py) made code changes to load feed_dict based on queue size. If not empty, load the feed_dict with actual queue values else load defined placeholders before passing to sess.run  | 
| *Dynamic decode (tacotron.py):* <br/> We observed that the tf.dynamic_deocde wrapper is not supported by NPU. Dynamic decode wrapper takes CustomDecoder as input and performs dynamic decoding at each step  | We found tf.while_loop as the replacement for tf.dynamic_decode and was supported by NPU. Implemented tf.while_loop consisting of condition and body functions. The body function outputs the next step, next inputs, outputs, final output and next state. Frames prediction and final decoder state were obtained as outputs from tf.while_loop  | 
| *Dynamic decode (dynamic_decode_test.py):* <br/> Tf.while_loop successfully worked on NPU but degraded the training performance (~higher steps/sec)  | We modified the tf.dynamic_decode souce code. Replaced the source code logic with our tf.while_loop implementation as stated above. Tested this approach on NPU and drastically improved the training performance  |
| *Tensor Slicing in TacoTestHelper (helpers.py):* <br/> NPU does not support pythonic way of slicing tensors. For example: next_inputs = outputs[:, -self._output_dim:] | Replaced next_inputs = outputs[:, -self._output_dim:] as next_inputs = tf.slice(outputs, [0, tf.shape(outputs)[1] -self._output_dim], [self._hparams.tacotron_batch_size, self._output_dim])| 
| *Unsupported tf.py_func operator during Model conversion (change_graph.py):* <br/> tf.py_func takes inputs, split_func function and output float type as inputs and wraps it in the tf graph | Removed py_func operator and logic from the model. Used split_func to get correct tensor dimensions for the placeholders and then used them as input to the model | 


 ### Training Performance (NPU vs GPU): 
 
 - We made the above mentioned code changes after using code conversion tool to train on NPU. While training we observed the performance was slower than on GPU (~ x10 slower). 
 
 - We used the [Profiling tool](https://gitee.com/ascend/modelzoo/wikis/%E8%AE%AD%E7%BB%83%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E6%8C%87%E5%BC%95(CANN3.2)?sort_id=3652440) to observe which operators were taking the maximum time to execute and also raised following issues on gitee.
 
 - We then used *allow_mix_precision* while training on NPU. *allow_mix_precision:* According to the built-in optimization strategy, the accuracy of some float32 operators can be automatically reduced to float16 for operators of the float32 data type in the entire network, thereby improving system performance and reducing memory usage with a small loss of accuracy. This drastically improved the training performance and had the following observation:
 
 
|  |  Average GPU Training (sec/step) | Average NPU Training (sec/step) |
| --------  | --------  | ------------------- |
| Without allow_mix_precision  | 1.09 | 13.45 |  
| With allow_mix_precision  | 1.09 | 1.07 |  

 *allow_mix_precision:* does not apply to GPU training

### Issues raised on Gitee community:
 
- https://gitee.com/ascend/modelzoo/issues/I3VEXH?from=project-issue
 
- https://gitee.com/ascend/modelzoo/issues/I41PZR?from=project-issue
