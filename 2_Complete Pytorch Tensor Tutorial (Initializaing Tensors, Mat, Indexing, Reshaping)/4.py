import torch

# ==============================
#       Tensor Reshaping        
# ==============================

x = torch.arange(9)

x_3x3 = x.view(3, 3) # Reshape x into a 3x3
print(x)

x_3x3 = x.reshape(3, 3) # Similar to what view does
print(x)

y = x_3x3.t() # Transposing a tensor
print(y)

x_1 = torch.rand((2, 5))
x_2 = torch.rand((2, 5))
print(torch.cat((x_1, x_2), dim=0).shape) # Concatinating tensors along a dimension

z = x_1.view(-1) # Flattening the tensor
print(x.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) # Flatten the tensor but not the batch
print(z.shape)

z = x.permute(0, 2, 1) # Swaping dimensions of the tensor
print(z.shape)

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(dim=0).shape) # Adding a dimension to the tensor
print(x.unsqueeze(dim=1).shape)

x = torch.arange(10).unsqueeze(dim=0).unsqueeze(dim=1)
print(x.shape)
print(x.squeeze(dim=1).shape) # Remove a dimension of a tensor