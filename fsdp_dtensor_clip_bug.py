import os
import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.distributed.tensor import init_device_mesh, distribute_tensor, Shard, Replicate, Partial
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
    
    # Construct a tensor where rows are distinct per rank
    global_tensor = torch.stack([torch.tensor([1.0, 1.0])*i for i in range(1, world_size * 1 + 1)])  # shape: (1 * world_size, 4)

    # Shard along dim=0: each rank gets 1 row
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)]) 
    print(f"[Rank {rank}] Local tensor: {dtensor.to_local()}")
    
    dTensorNorm = torch.sum(dtensor.to_local()**2.0)
    print(f"[Rank {rank}] dTensorNorm: {dTensorNorm.item():.10f}")
    all_reduce(dTensorNorm, op=ReduceOp.SUM)
    print(f"[Rank {rank}] dTensorNormAfterReduce: {dTensorNorm.item():.10f}")
    calcNorm = dTensorNorm**(1/2)
    print(f"[Rank {rank}] calcNorm: {calcNorm.item():.10f}")

    tensorNorm = torch.linalg.vector_norm(global_tensor)
    if rank == 0:
        print(f"TensorNorm: {tensorNorm.item():.10f}")

if __name__ == "__main__":
    main()
