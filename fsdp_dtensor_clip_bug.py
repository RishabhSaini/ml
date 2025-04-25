import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor
from torch.nn.utils.clip_grad import _get_total_norm

def setup():
    init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", (world_size,))
    return rank, world_size, mesh

def cleanup():
    destroy_process_group()

def main():
    rank, world_size, mesh = setup()
    
    MAX_NORM = 1.0
    EPSILON = 1e-6

    # Construct a tensor where rows are distinct per rank
    global_tensor = torch.stack([torch.ones(1) * (i/10.0 + 0.1) for i in range(world_size * 1)])  # shape: (1 * world_size, 4)

    # Shard along dim=0: each rank gets 1 row
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)]) 
    print(f"[Rank {rank}] Local tensor: {dtensor.to_local()}")
   
    param = torch.nn.Parameter(dtensor)  # wrap DTensor as Parameter
    param.grad = dtensor.clone()         # manually set grad

    dTensorNorm = _get_total_norm([dtensor]) + 1.0 - 1.0
    print(f"[Rank {rank}] dTensorNorm: {dTensorNorm.item():.10f}, {dTensorNorm.placements}")

    tensorNorm = _get_total_norm(global_tensor)
    if rank == 0:
        print(f"TensorNorm: {tensorNorm.item():.10f}")

if __name__ == "__main__":
    main()
