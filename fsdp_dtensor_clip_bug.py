import os
import torch
from torch.distributed import init_process_group
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor
#from torch.distributed.tensor.experimental import implicit_replication

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

    param = torch.nn.Parameter(torch.randn(2*world_size, 4))  # wrap DTensor as Parameter
    param.grad = dtensor.clone()         # manually set grad
    grad_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
    print(f"[Rank {rank}] Grad norm: {grad_norm.item():.6f}, Param Grad: {param.grad.to_local()}")

    norm_shard = torch.linalg.vector_norm(dtensor, 2)
    with_eps = 1.0 / (norm_shard + 1e-6)
    actual_norm = 1.0/with_eps - 1e-6
    print(f"[Rank {rank}] norm per shard: {norm_shard.item():.8f}, coeff: {with_eps.item():.10f}, global norm: {actual_norm.item():.8f},")
    
    actual_norm_local = torch.linalg.vector_norm(dtensor, 2) 
    norm_shard_local = torch.linalg.vector_norm(dtensor.to_local(), 2)
    with_eps_local = 1.0 / (norm_shard_local + 1e-6)
    print(f"[Rank {rank}] norm per shard local: {norm_shard_local.item():.8f}, coeff_local: {with_eps_local.item():.10f}, global norm local: {actual_norm_local.item():.8f},")
    
if __name__ == "__main__":
    main()
