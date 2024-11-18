import torch

# ========================================
#   Tensor Math & Comparison Operations
# ========================================

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)

z3 = x + y
print(z3)

# Subtraction
z1 = torch.empty(3)
torch.subtract(x, y, out=z1)
print(z1)

z2 = torch.sub(x, y)
print(z2)

z3 =x - y
print(z3)

# Multiplication
z1 = torch.empty(3)
torch.mul(x, y, out=z1)
print(z1)

z2 = torch.mul(x, y)
print(z2)

z3 = x * y
print(z3)

# Division
z1 = torch.empty(3)
torch.div(x, y, out=z1)
print(z1)

z2 = torch.div(x, y)
print(z2)

z3 = x / y
print(z3)

# Exponentiation
z1 = torch.empty(3)
torch.pow(x, y, out=z1)
print(z1)

z2 = torch.pow(x, y)
print(z2)

z3 = x ** y
print(z3)

# Dot product
z2 = torch.dot(x, y)
print(z2)


# Simple Comparison
z1 = x > 0
print(z1)

z2 = x < 0
print(z2)

# Matrix Multiplication
x = torch.rand((2, 5))
y = torch.rand((5, 3))
z = torch.mm(x, y) # or x.mm(y)
print(z)

# Matrix Exponentiation
x = torch.rand((5, 5))
print(x.matrix_power(3))

# Batch Matrix Multiplication
batch = 32
n = 18
m = 20
p = 30
tensor_1 = torch.rand((batch, n, m))
tensor_2 = torch.rand((batch, m, p))
out_batch_multiplication = torch.bmm(tensor_1, tensor_2)
print(out_batch_multiplication)

# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2 # This will extend x2 to match the shape of x1
print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0) # Summation along a dimension
print(sum_x)

values, indeces = torch.max(x, dim=0) # Find max along col
print(values, indeces)

values, indeces = torch.min(x, dim=0) # Find min along col
print(values, indeces)

abs_x = torch.abs(x) # Compute Absolute value of x
print(abs_x)

z = torch.argmax(x, dim=0) # Find max index along col
print(z)

z = torch.argmin(x, dim=1) # Find min index along row
print(z)

mean_x = torch.mean(x, dim=1) # Compute mean along col
print(mean_x)

z = torch.eq(x1, x2) # Check equality
print(z)

sorted_y, indeces = torch.sort(y, dim=0, descending=False) # Sort elemnts
print(sorted_y)

z = torch.clamp(x, min=0, max=10) # Clamp torch elements to min and or max
print(z)

k = torch.tensor([1, 0, 1, 0], dtype=bool)
l = torch.any(k) # Returns True if any element is True
print(l)

m = torch.all(k) # Returns True if all elements are True
print(m)