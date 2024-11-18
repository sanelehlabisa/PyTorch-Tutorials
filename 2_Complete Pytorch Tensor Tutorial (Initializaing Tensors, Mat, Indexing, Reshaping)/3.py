import torch

# ===================================
#           Tensor Indexing          
# ===================================

batch_size = 10
feactures = 25
x = torch.rand((batch_size, feactures))

print(x[0].shape) # Get all features in first row

print(x[:, 0].shape) # Get all batch_sizes in first column

print(x[2, 0:10]) # 0:10 -> [0, 1, 2, ..., 9]

x[0, 0] = 100 # Element assignment
print(x)

# Fancy indexing
x = torch.arange(10)
indeces = [2, 5, 8]
print(x[indeces]) # pick features at prvided indeces

x = torch.rand((3, 5))
print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # Picking elements at given rows and cols

x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # Pick elements that cetify the condition
print(x[x.remainder(2) == 0]) # Pick elements that are divisible by 2

print(torch.where(x > 5, x, x + 2)) # Print x where condition is met else do something else

print(torch.tensor([0, 0, 1, 2, 4, 4, 2, 5, 6]).unique()) # Remove duplicate

print(x.ndimension()) # Get dimenstions of tensor

print(x.numel()) # Count elemnts in a tensor
