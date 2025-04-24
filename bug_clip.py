import os
import torch
import torch.distributed as dist

def setup():
    dist.init_process_group("gloo")  # or "nccl" for GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def clip_with_bug(local_norm, max_norm):
    # Simulates the buggy line: adds 1e-6 per rank
    local_norm += 1e-6
    total_norm = local_norm.clone()
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
    clip_coef = max_norm / total_norm
    return clip_coef

def clip_corrected(local_norm, max_norm):
    # Add epsilon after total norm is computed
    total_norm = local_norm.clone()
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
    clip_coef = max_norm / (total_norm + 1e-6)
    return clip_coef

def main():
    rank, world_size = setup()
    # Let's say each rank has a local norm of 2.0
    local_norm = torch.tensor(2.0)
    max_norm = 1.0

    buggy_coef = clip_with_bug(local_norm.clone(), max_norm)
    correct_coef = clip_corrected(local_norm.clone(), max_norm)

    if rank == 0:
        print(f"[Rank {rank}] Buggy clip coef:    {buggy_coef.item()}")
        print(f"[Rank {rank}] Correct clip coef:  {correct_coef.item()}")

    cleanup()

if __name__ == "__main__":
    # Run this with 4 ranks:
    # torchrun --nproc-per-node=4 bug_clip.py
    main()
