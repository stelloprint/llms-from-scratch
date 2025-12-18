# From Appendix A: Introduction to PyTorch
# https://livebook.manning.com/book/build-a-large-language-model-from-scratch/appendix-a

# Listing A.1 Creating PyTorch tensors
# Note: matrices and above are indented to make them easier to read.

import torch

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2], 
                                     [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], 
                                     [[5, 6], [7, 8]]])

print(tensor0d.dtype) # dtype: torch.int64
print(tensor1d)
print(tensor2d)
print(tensor3d)
print(tensor3d.shape)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype) # dtype: torch.float32
print(floatvec)

floatvec2 = tensor3d.to(torch.float32)
print(floatvec2.dtype) # dtype: torch.float32
print(floatvec2)

