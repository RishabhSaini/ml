import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

linear_1 = nn.Linear(2, 2)
with torch.no_grad():
    linear_1.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    linear_1.bias.copy_(torch.tensor([0.5, 0.6]))

linear_2 = nn.Linear(2, 1)
with torch.no_grad():
    linear_2.weight.copy_(torch.tensor([0.7, 0.8]))
    linear_2.bias.copy_(torch.tensor([0.9]))

# Define a simple model
model = nn.Sequential(
    linear_1,
    nn.ReLU(),
    linear_2
)

# Dummy input and target
x = torch.tensor([[0.11, 0.12]])
target = torch.tensor([[0.13]])

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Forward + backward
output = model(x)
loss = criterion(output, target)
loss.backward()

before = []
for i, p in enumerate(model.parameters()):
    #print(f"Param: {i}\n Weight: {p},\n Gradient: {p.grad},\n Norm: {p.grad.norm()}\n")    
    before.append(p.grad.norm())

# Clip gradients
max_norm = 1.0
total_norm = clip_grad_norm_(model.parameters(), max_norm)
clamped_clip_coeff = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
print(f"Total Norm: {total_norm}, Max Norm: {max_norm}, Clipping Coeff: {clamped_clip_coeff}")

after = []
for i, p in enumerate(model.parameters()):
    #print(f"Param: {i}\n Weight: {p},\n Gradient: {p.grad},\n Norm: {p.grad.norm()}\n")    
    after.append(p.grad.norm())

print(f"Gradient Before:{before}")
print(f"Gradient After :{after}")

# Optimizer step
optimizer.step()
