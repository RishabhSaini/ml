import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor

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
    global_tensor = torch.stack([torch.ones(4) * (i/10.0 + 0.1) for i in range(world_size * 2)])  # shape: (2 * world_size, 4)

    # Shard along dim=0: each rank gets 2 rows
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)]) 
    print(f"[Rank {rank}] Local tensor:\n{dtensor.to_local()}")
   
    param = torch.nn.Parameter(dtensor)  # wrap DTensor as Parameter
    param.grad = dtensor.clone()         # manually set grad

    total_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=MAX_NORM)
    clip_coef = MAX_NORM / (total_norm + EPSILON)
    print(f"[Rank {rank}] Partial norm: {total_norm.item():.10f}, {total_norm.placements}, clip_coef: {clip_coef.item():.10f}, {clip_coef.placements}")

    local_param = torch.nn.Parameter(global_tensor)  # wrap DTensor as Parameter
    local_param.grad = global_tensor.clone()         # manually set grad

    local_total_norm = torch.nn.utils.clip_grad_norm_([local_param], max_norm=MAX_NORM)
    local_clip_coef = MAX_NORM / (local_total_norm + EPSILON)
    epsScale = MAX_NORM / (clip_coef*EPSILON) - local_total_norm / EPSILON
    
    if rank == 0:
        print(f"     Local total norm: {local_total_norm.item():.10f},                                            Local clip_coef: {local_clip_coef.item():.10f}, epsScale: {epsScale.item():.10f}")

if __name__ == "__main__":
    main()
