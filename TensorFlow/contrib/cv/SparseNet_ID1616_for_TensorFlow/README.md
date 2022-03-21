# SparseNet
Sparsely Aggregated Convolutional Networks [[PDF](https://arxiv.org/abs/1801.05895)]

[Ligeng Zhu](https://lzhu.me), [Ruizhi Deng](http://www.sfu.ca/~ruizhid/), [Michael Maire](http://ttic.uchicago.edu/~mmaire/), [Zhiwei Deng](http://www.sfu.ca/~zhiweid/), [Greg Mori](http://www.cs.sfu.ca/~mori/), [Ping Tan](https://www.cs.sfu.ca/~pingtan/)


# Requirements
```
opencv-python==4.5.3.56 or 3.4.9.31
python==3.7.5 or 3.7.11
tensorflow-gpu==1.15.0
tensorpack==0.11
```

# Dataset
CIFAR10

<table> 
<th>Data augmentation: 
</th>
<td>standard
<td>mirroring
<td>shifting
</table>

<table> 
<th>Preprocessing: </th>
<td>normalize the data by the channel mean and standard deviation

</table>

# Result
<table>
<th>GPU-Result:</th>
<td>requirements.txt
<td>graph.pbtxt
<td>loss+perf_gpu.txt
<td>checkpoint_gpu
</table>
<table>
<th>NPU-Result:</th>
<td>requirements.txt
<td>graph.pbtxt
<td>loss+perf_npu.txt
<td>checkpoint_npu
</table>

# Performance

<table> 

<th> 

Platform | Architecture | Depth | Params | CIFAR10 | Training(epoch) | Inference(epoch)
--- | --- | --- | --- | --- | --- | --- |
Paper |SparseNet (k=24)  | 100 | 2.5M | 4.44 | --- | ---
--- | --- | --- | --- | --- | --- | --- |
GPU |SparseNet (k=24)  | 100 | 2.5M | 5.0 | 1 minute 1 seconds ~ 1 minute 10 seconds | 3.8 seconds ~ 4.1 seconds
NPU |SparseNet (k=24)  | 100 | 2.5M | 4.8 | 1 minute 0 seconds ~ 1 minute 1 seconds |  3.4 seconds ~ 3.5 seconds
 </th>
 </table>


# Train on Dataset
<table>
<th> CIFAR10-GPU </th>
<th> run_1p.sh </th>
<td>python ./main_gpu.py --dataset c10 --dataset_dir /cache/dataset/cifar10_data --fetch sparse --depth 100 --growth_rate 24 --batch_size 64 --drop_1 150 --drop_2 225 --max_epoch 300
</td>
</table>

<table>
<th> CIFAR10-NPU </th>
<th> train_1p_ci.sh </th>
<td>python ./main_npu.py --dataset c10 --dataset_dir /cache/dataset/cifar10_data --fetch sparse --depth 100 --growth_rate 24 --batch_size 64 --drop_1 150 --drop_2 225 --max_epoch 300
</td>
</table>


# SparseNet implementation


```
@article{DBLP:journals/corr/abs-1801-05895,
  author    = {Ligeng Zhu and
               Ruizhi Deng and
               Michael Maire and
               Zhiwei Deng and
               Greg Mori and
               Ping Tan},
  title     = {Sparsely Aggregated Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1801.05895},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.05895},
  archivePrefix = {arXiv},
  eprint    = {1801.05895},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-05895},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
