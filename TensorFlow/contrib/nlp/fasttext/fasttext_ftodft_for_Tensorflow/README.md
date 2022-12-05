# FToDTF - FastText on Distributed TensorFlow

This software uses unsupervised machine-learning to calculate vector-representation of words. These vector representations can then be used for things like computing the similarity of words to each other or association-rules (e.g. paris is to france like X to germany).

This software is an implementation of https://arxiv.org/abs/1607.04606 (facebook's fasttext) in tensorflow on Ascend 910 environment.  

In contrast to the original implementation of fasttext (https://github.com/facebookresearch/fastText) this implementation can use GPUs to accelerate the training and the training can be distributed across multiple nodes.

## Datasets
Any language corpus according to the supported languages mentioned in the fasttext original paper
Dataset has to be preprocessed before training

## Running
```
python3 cli.py preprocess --corpus_path <your-training-data>
python3 cli.py train (use `--` to add training parameters)
```  
in your console.  
This will run the training and will periodically store checkpoints of the current model into the ./log folder.
After you have trained for some time you can try out the trained word-vectors:
```
python3 cli.py infer similarity i you one two
```
This will load the latest model stored in ./log and use it to calculate and print the similarity between the words i you one two. If everything works out, "I" should be similar to "you" and "one" should be similar to "two", while all other combinations should be relatively un-similar.

## Docker
This application is also available as pre-built docker-image (https://hub.docker.com/r/dbaumgarten/ftodtf/)
```
sudo docker run --rm -it -v `pwd`:/data dbaumgarten/ftodtf train
```

## Distributed Setup
### Docker
There is docker-compose file demonstrating the distributed setup op this programm. To run a cluster on your local machine 
- go to the directory of the docker-compose file
- preprocess your data using `python3 cli.py preprocess --corpus_path <your-training-data>`
- run:
```
sudo docker-compose up
```
This will start a cluster consisting of two workers and two parameter servers on your machine.  
Each time you restart the cluster it will continue to work from the last checkpoint. If you want to start from zero delete the contents of ./log/distributed on the server of worker0
Please note that running a cluster on a single machine is slower then running a single instance directly on this machine. To see some speedup you will need to use multiple independent machines.
### Slurm
There is also an example how to use slurm for setting up distributed training (slurmjob.sh). You will probably have to modify the script to work on your specfic cluster. Please not that the slurm-script currently only handles training. You will have to create training-batches (fasttext preprocess) and copy the created batches-files to the cluster-nodes manually befor starting training.

## Training-data
The input for the proprocess-step is a raw text-file containing lots of sentences of the language for that you want to compute word-embeddings.

## Hyperparameters and Quality
The quality of the calculated word-vectors depends heavily on the used training-corpus and the hyperparameters (training-steps, embedding-dimension etc.). If you don't get usefull results try changing the default hyperparameters (especially the amount of training-steps can have a big influence) or use other training data.  

We got really good results for german with 81MB of training-data and the parameters --num_buckets=2000000 --vocabulary_size=200000 --steps=10000000, but the resulting model is quite large (2.5GB) and it took >10 hours to train.

## Known Bugs and Limitations
- When supplying input-text that does not contain sentences (but instead just a bunch of words without punctuation) ```fasttext preprocess``` will hang indefinetly.

## 训练结果

- 精度结果比对

|  精度指标项  | GPU/论文实测  | NPU实测 |
|  ----  | ----  | ---- |
| LOSS  | 2.5644302148967983 | 2.5319034051969647 |

- 性能结果比对

|  性能指标项  | GPU/论文实测  | NPU实测 |
|  ----  | ----  | ---- |
|  Steptime  | 0.004117727279663086 | 0.0055217742919921875 |
