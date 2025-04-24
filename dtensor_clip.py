import os
import torch
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor

mesh = init_device_mesh("cpu", (int(os.environ["WORLD_SIZE"]),))
big_tensor = torch.randn(4, 4)
# Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
result = my_dtensor + 1e-1

print(f"BigTensor: {my_dtensor} \nResult: {result}")
