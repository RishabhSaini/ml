import os
import torch
from torch.distributed import init_process_group, all_reduce, ReduceOp
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor

def setup():
    init_process_group(backend="gloo")

def compute_global_norm(tensor):
    local_norm_sq = torch.sum(tensor ** 2)
    all_reduce(local_norm_sq, op=ReduceOp.SUM)
    return local_norm_sq.sqrt()

def main():
    setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    mesh = init_device_mesh("cpu", (world_size,))

    # Construct a tensor where rows are distinct per rank
    global_tensor = torch.stack([
        torch.ones(4) * (i + 1) for i in range(world_size * 2)
    ])  # shape: (2 * world_size, 4)

    # Shard along dim=0: each rank gets 2 rows
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])
    
    # compute correct global norm
    global_norm = compute_global_norm(dtensor.to_local())
    correct_clip_coef = 1.0 / (global_norm + 1e-6)
    print(f"[Rank {rank}] Global norm: {global_norm.item():.6f}, correct clip coef: {correct_clip_coef.item():.6f}")

    # Each rank gets different data → different norm
    local_norm = dtensor.norm()
    clip_coef = 1.0 / (local_norm + 1e-6)

    print(f"[Rank {rank}] Local tensor:\n{dtensor.to_local()}")
    print(f"[Rank {rank}] Local norm: {local_norm.item():.6f}, clip coef: {clip_coef.item():.6f}")

if __name__ == "__main__":
    main()
