import numpy as np
import torch
from torch.nn.functional import conv2d

input = torch.from_file("input.bin", size=1 * 3 * 224 * 224, dtype=torch.float32)
input = input.resize(1, 3, 224, 224)
filter = torch.from_file("filter.bin", size=32 * 3 * 3 * 3, dtype=torch.float32)
filter = filter.resize(32, 3, 3, 3)

stride = [2, 2]
padding = [1, 1]

output = conv2d(input, filter, stride=stride, padding=padding)
output_np = output.numpy()

output_npu_np = np.fromfile("output_npu.bin", dtype=np.float32).reshape(1, 32, 112, 112)
print(np.allclose(output_npu_np, output_np, atol=1e-05, rtol=1e-05))
