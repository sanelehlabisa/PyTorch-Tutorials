# ==============================
#       Einsum in PyTorch       
# ==============================

import torch

x = torch.rand((2, 3))

# Permutation of Tensor
print(torch.einsum("ij->ji", x))

# Sumation
print(torch.einsum("ij->", x))

# Column sum
print(torch.einsum("ij->j", x))

# Row sum
print(torch.einsum("ij->i", x))

# Matrix Vector Multiplication
v = torch.rand((1, 3))
print(torch.einsum("ij,kj->ik", x, v))

# Matrix Matrix Multiplication
print(torch.einsum("ij,kj->ik", x, x))

# Dot product first row with first row of matrix
print(torch.einsum("i,i->", x[0], x[0]))

# Dot product with matrix
print(torch.einsum("ij,ij->", x, x))

# Handmard product (element-wise multiplication)
print(torch.einsum("ij,ij->ij", x, x))

# Outer product
a = torch.rand((3))
b = torch.rand((5))
print(torch.einsum("i,j->ij", a, b))

# Batch Matrix Multiplication
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
print(torch.einsum("ijk,ikl->ijl", a, b))

# Matrix Diagonal
x = torch.rand((3, 3))
print(torch.einsum("ii->i", x))

# Matrix Trace
print(torch.einsum("ii->", x))