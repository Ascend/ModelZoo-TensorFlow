import re
import numpy as np

#NPU上运行的日志文件下载到本地的路径
path = ""

with open(path, mode='r') as f:
    NpulogInfo = f.readlines()

NpulogInfo = str(NpulogInfo)
valuesNPU = re.findall(r"global_step/sec:(.*?)\\n", NpulogInfo)
valuesNPU = [float(x) for x in valuesNPU]

perf_npu = np.mean(valuesNPU)
print(perf_npu)
