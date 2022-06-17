import numpy as np

import numpy as np
#生成64*128随机噪声数据,类型float32
data = np.random.uniform(size=(64,128))
data = data.astype(np.float32)
data.tofile("bin_data/infer_bin_data.bin")