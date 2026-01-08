# From Appendix A: Introduction to PyTorch
# https://livebook.manning.com/book/build-a-large-language-model-from-scratch/appendix-a

# Understanding Tensors

# Listing A.1 Creating PyTorch tensors
# Note: matrices and above are indented to make them easier to read.

import torch

# Create tensors of different dimensions

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2], 
                                     [3, 4]])
                                     
tensor3d = torch.tensor([[[1, 2], [3, 4]], 
                                     [[5, 6], [7, 8]]])

#  Tensor Data Types

# This choice is primarily due to the balance between precision and computational efficiency. 
# A 32-bit floating-point number offers sufficient precision for most deep learning tasks while 
# consuming less memory and computational resources than a 64-bit floating-point number. 
# Moreover, GPU architectures are optimized for 32-bit computations, and using this data type 
# can significantly speed up model training and inference.

print('tensor0d.dtype:', tensor0d.dtype) # dtype: torch.int64
print('tensor1d:', tensor1d)
print('tensor2d:', tensor2d)
print('tensor3d:', tensor3d)
print('tensor3d.shape:', tensor3d.shape)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print('floatvec.dtype:', floatvec.dtype) # dtype: torch.float32
print('floatvec:', floatvec)


# it is possible to change the precision using a tensor’s .to method. The following code demonstrates this 
# by changing a 64-bit integer tensor into a 32-bit float tensor

floatvec2 = tensor3d.to(torch.float32)
print('floatvec2.dtype:', floatvec2.dtype) # dtype: torch.float32
print('floatvec2:', floatvec2)

# Common Pytorch Tensor Operations

# Create Tensor
tensor2d = torch.tensor([[1, 2, 3], 
                         [4, 5, 6]])

print('tensor2d:', tensor2d) # Output: tensor([[1, 2, 3], [4, 5, 6]])

# .shape returns [2, 3], meaning the tensor has two rows and three columns
print('tensor2d.shape:', tensor2d.shape) # Output: torch.Size([2, 3])

# To reshape the tensor into a 3 × 2 tensor, we can use the .reshape method
print('tensor2d.reshape(3, 2):', tensor2d.reshape(3, 2)) # Output: tensor([[1, 2], [3, 4], [5, 6]])

# The more common command for reshaping tensors in PyTorch is .view():
print('tensor2d.view(3, 2):', tensor2d.view(3, 2)) # Output: tensor([[1, 2], [3, 4], [5, 6]])

# Similar to .reshape and .view, in several cases, PyTorch offers multiple syntax options for executing 
# the same computation. PyTorch initially followed the original Lua Torch syntax convention but then, by 
# popular request, added syntax to make it similar to NumPy. (The subtle difference between .view() and 
# .reshape() in PyTorch lies in their handling of memory layout: .view() requires the original data to be 
# contiguous and will fail if it isn’t, whereas .reshape() will work regardless, copying the data if necessary to 
# ensure the desired shape.)

# Next, we can use .T to transpose a tensor, which means flipping it across its diagonal. 
# Note that this is not the same as reshaping a tensor, as you can see based on the following result:

print('tensor2d.T:', tensor2d.T)

# The output is tensor([[1, 4], [2, 5], [3, 6]])

# Lastly, the common way to multiply two matrices in PyTorch is the .matmul method:

print('tensor2d.matmul(tensor2d.T):', tensor2d.matmul(tensor2d.T))

# The output is tensor([[14, 32], [32, 77]])

# However, we can also adopt the @ operator, which accomplishes the same thing more compactly:

print('tensor2d @ tensor2d.T:', tensor2d @ tensor2d.T)

# The output is tensor([[14, 32], [32, 77]])
