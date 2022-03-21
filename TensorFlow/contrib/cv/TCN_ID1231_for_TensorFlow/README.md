# TF TCN
*Tensorflow Temporal Convolutional Network*

This is an implementation of [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) in TensorFlow.

# About this code
This repo contains code that can run on Ascend 910, which is convered from the original repo: https://github.com/YuanTingHsieh/TF_TCN . 


# Domains and Datasets

  - Copying Memory Task with various T (we evaluated on T=500, 1000, 2000)

    Generate data for the copying memory task：
        """
        :param T: The total blank time length
        :param mem_length: The length of the memory to be recalled
        :param b_size: The batch size
        :return: Input and target data tensor
        """
          def data_generator(T, mem_length, b_size):
            seq = np.random.randint(1, 9, size=(b_size, mem_length))
            zeros = np.zeros((b_size, T))
            marker = 9 * np.ones((b_size, mem_length + 1))
            placeholders = np.zeros((b_size, mem_length))
    
            x = np.concatenate((seq, zeros[:, :-1], marker), 1)
            y = np.concatenate((placeholders, zeros, seq), 1)
    
            return x, y
     
# requirements
   pip install -r requirements.txt

# Run
1. Create a notebook environment with Ascend 910 on ModelArts.
2. Upload this folder to the work dir.
3. In the corresponding directory of this repo, type the following to run  experiment
```
                                     python3  [module_name]
in "copymem":               python3  copymem.copymem_test.py

```
# Performance
Pricision:
|Dataset | On GPU | On Ascend 910 |
| :----: | :----: | :----: |
|Generate data| 100 | 100 |

## References
[1] Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." arXiv preprint arXiv:1803.01271 (2018).
[2] Salimans, Tim, and Diederik P. Kingma. "Weight normalization: A simple reparameterization to accelerate training of deep neural networks." Advances in Neural Information Processing Systems. 2016.
