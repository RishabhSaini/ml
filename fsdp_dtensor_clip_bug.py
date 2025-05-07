import os
import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.distributed.tensor import init_device_mesh, distribute_tensor, Shard, Replicate, Partial
from torch.nn.utils.clip_grad import _get_total_norm
from torch._dynamo import optimize
import operator

import sys
def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        print(f"Called {func_name}() in {filename}:{lineno}")
    return trace_calls

def setup():
    init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", (world_size,))
    return rank, world_size, mesh

def cleanup():
    destroy_process_group()

def is_reduction_op(node):
    return node.op == 'call_function' and node.target in {
        operator.add, operator.sub, operator.mul, operator.truediv
}

def my_compiler(graph_module: torch.fx.GraphModule, _):
    print(graph_module)
    for node in graph_module.graph.nodes:
        if node.target == torch.linalg.vector_norm:
            # Check if next users include a reduction op
            for user in node.users:
                if is_reduction_op(user):
                    print(f"Detected vector_norm followed by reduction: {user.op} {user.target}")
                    # You can now inject logic for DTensor conversion
    return graph_module.forward

@optimize(my_compiler)
def getNorm(x):
    norm = torch.linalg.vector_norm(x)
    return norm * 1.0

def main():
    rank, world_size, mesh = setup()
    
    # Construct a tensor where rows are distinct per rank
    global_tensor = torch.stack([torch.tensor([1.0, 1.0])*i for i in range(1, world_size * 1 + 1)])  # shape: (1 * world_size, 2)

    # Shard along dim=0: each rank gets 1 row
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)]) 
    print(dtensor) 
    #dTensorNorm = torch.sum(dtensor.to_local()**2.0)
    #all_reduce(dTensorNorm, op=ReduceOp.SUM)
    #calcNorm = dTensorNorm**(1/2)
    #givenNorm = _get_total_norm(dtensor) * 1.0
    tensorNorm = getNorm(dtensor)
    
    if rank == 0:
        #sys.settrace(trace_calls)
        #shardNorm = torch.linalg.vector_norm(dtensor)
        #sys.settrace(None)
        #print(f"_get_total_norm: {givenNorm.item():.10f}, CorrectNorm: {tensorNorm.item():.10f}, calculatedNorm: {calcNorm.item():.10f}")
        print(f"_get_total_norm: {tensorNorm.item():.10f}")

if __name__ == "__main__":
    main()
