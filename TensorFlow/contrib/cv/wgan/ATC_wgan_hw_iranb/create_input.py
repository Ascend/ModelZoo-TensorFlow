import numpy as np

# wgan 输入为一个噪声向量


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])


# sheng cheng zdim = 100 的噪声向量保存到bin
for i in range(100):
    z_sampler = NoiseSampler()(1, 100)
    z_sampler.tofile("./input_bin/input_noise_{}.bin".format(i))
