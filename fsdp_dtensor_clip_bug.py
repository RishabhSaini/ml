import os
import torch
from torch.distributed import init_process_group
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor

def setup():
    init_process_group(backend="gloo")

def main():
    setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    mesh = init_device_mesh("cpu", (world_size,))

    # Construct a tensor where rows are distinct per rank
    global_tensor = torch.stack([
        torch.ones(4) * (i/10.0 + 0.1) for i in range(world_size * 2)
    ])  # shape: (2 * world_size, 4)

    # Shard along dim=0: each rank gets 2 rows
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)]) 
    print(f"[Rank {rank}] Local tensor:\n{dtensor.to_local()}")
   
    param = torch.nn.Parameter(dtensor)  # wrap DTensor as Parameter
    param.grad = dtensor.clone()         # manually set grad

    total_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
    print(f"[Rank {rank}] Total norm: {total_norm.item():.6f}, {total_norm.placements}")
    
    clip_coef = 1.0 / (total_norm + 1e-3)
    print(f"[Rank {rank}] clip_coef: {clip_coef.item():.6f}, {clip_coef.placements}")

if __name__ == "__main__":
    main()
