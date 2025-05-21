import torch

tensor_2d = torch.rand(3, 4)
tensor_3d = torch.zeros(2, 3, 4)

print(tensor_2d)
print(tensor_3d)

print(tensor_2d.dtype)
print(tensor_3d.dtype)

my_tensor = torch.arange(10)
print(my_tensor)

reshaped_tensor = my_tensor.reshape(2, 5)
print(reshaped_tensor)

reviewed_tensor = my_tensor.view(5, 2)
print(reviewed_tensor)

my_tensor[0] = 42

print(reshaped_tensor)
print(reviewed_tensor)

print(reviewed_tensor[:, 1])


tensor_a = torch.tensor([1, 2, 3, 4])
tensor_b = torch.tensor([5, 6, 7, 8])
# add two tensors (element-wise addition)
print(tensor_a + tensor_b)
print(torch.add(tensor_a, tensor_b))
# subtract two tensors (element-wise subtraction)
print(tensor_b - tensor_a)
print(torch.sub(tensor_b, tensor_a))
# multiplication (element-wise multiplication)
print(tensor_a * tensor_b)
print(torch.mul(tensor_a, tensor_b))
# Division (element-wise, converts to float)
print(tensor_b / tensor_a)
print(torch.div(tensor_b, tensor_a))
# remainders (element-wise)
print(tensor_b % tensor_a)
print(torch.remainder(tensor_b, tensor_a))
# exponents / power (element-wise)
print(torch.pow(tensor_a, tensor_b))
# reassignment of tensors 
tensor_a = tensor_a + tensor_b # longhand
tensor_a.add_(tensor_b) #inplace add
