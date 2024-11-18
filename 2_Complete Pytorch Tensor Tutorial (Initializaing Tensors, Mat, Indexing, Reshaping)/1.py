import torch

# ===================================
#       Tensor Initialization
# ===================================

# Set device to cuda if gpu is availble else set to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a tensor from list and define datatype and device
my_tensor =  torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

# Print tensor value
print(my_tensor)

# Print tensor datatype
print(my_tensor.dtype)

# Print tensor device
print(my_tensor.device)

# Print tensor shape
print(my_tensor.shape)

# Print tensor requres gradient
print(my_tensor.requires_grad)


# Initialize empty tensor
x = torch.empty(size=(3, 3))
print(x)

# Initialize tensor of zeros
x = torch.zeros(size=(2, 2))
print(x)

# Initialze tensor of ones
x = torch.ones(size=(1, 2))
print(x)

# Initialize tensor of random numbers
x = torch.rand(size=(2, 2))
print(x)

# Initialize tensor of identity matrix
x = torch.eye(n=2)
print(x)

# Initialize tensor of range
x = torch.arange(start=0, end=5, step=1)
print(x)

# Initialize tensor of linespace
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

# Initialize empty tensor with normal distribution
x = torch.empty(size=(2, 2)).normal_(mean=0, std=1)
print(x)

# Initialize empty tensor with uniform distribution
x = torch.empty(size=(2, 2)).uniform_(0, 1)
print(x)

# Initialize diagonal tensor of ones
x = torch.diag(torch.ones(3))
print(x)


# Initialize tensor Convert tensor into bool (int, float, double)
tensor =  torch.arange(4)
print(tensor.bool()) # Convert tensor to bool
print(tensor.short()) # Convert tensor to short
print(tensor.long()) # Convert tensor to long
print(tensor.half()) # Convert tensor to half
print(tensor.float()) # Convert tensor to float
print(tensor.double()) # Convert tensor to double

# Array to tensor conversion and vice-versa
import numpy as np
np_array = np.ones((5, 2))
tensor = torch.tensor(np_array) # Convert numpy array into tensor
np_array = tensor.numpy()
print(tensor)
print(np_array)
