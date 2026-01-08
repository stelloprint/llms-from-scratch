# A.3 Seeing models as computation graphs

# Computational graphs are directed graphs representing the mathematical operations of a model.
# In Deep Learning, these graphs layout the sequence of calculations needed to compute the output of a neural network.
# The graph is used by PyTorch's autograd system to automatically compute gradients of model parameters, 
# meaning that it traces back the partial derivatives of each operation starting from the loss function resulting
# in gradients that can be used to update parameters, weights, and biases to improve the model's performance.

# A Logistic Regression Forward Pass

# Logistic Regression:
# A continuous number 'z' is predicted (regression) and then squashed to a probability between 0 and 1
# via the logistic (sigmoid) function in order to compare it to the true label (target) and calculate the loss.

import torch
import torch.nn.functional as F

# 1. Define the data and parameters
y = torch.tensor([1.0,]) # true label (target $y$ ground truth)
# In a real dataset, y would be a long vector of many labels (e.g., [1.0, 0.0, 1.0, 1.0...]). 
# Just keep in mind that the "shape" of your label tensor must always match the "shape" of your prediction tensor.
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2], requires_grad=True) # weight (learned parameter)
b = torch.tensor([0.0], requires_grad=True) # bias (learned parameter)


# 2. Calculate the "Net Input" (Linear Combination)
# In PyTorch, the value z (the raw output before sigmoid) is often called the "Logits."
# Because $z$ is the value that would result if you ran the Logit function on the final probability, 
# we call $z$ the "Logit." In Deep Learning, the convention has become: 
# "Logits = the raw, un-normalized scores produced by the last layer of a model."
z = x1 * w1 + b # input times weight plus bias (net input)
# z = (1.1 * 2.2) + 0.0 = 2.42
# Linear combination can be any number from negative infinity to positive infinity.
# The sigmoid function is a non-linear function that maps the linear combination 
# to a probability between 0 and 1.

# 3. Apply the Logistic Function (Sigmoid)
# This converts 2.42 into a probability between 0 and 1 (approx 0.918)
# In statistics, "Logit" is the inverse of the sigmoid function.
# 
a = torch.sigmoid(z) # activation function prediction ($\hat{y}$ aka "y-hat")

# 4. Calculate the Loss
# Compares the prediction (0.918) to the true label (1.0)
loss = F.binary_cross_entropy(a, y) # loss function
# Simple substraction is not used because Binary Cross Entropy (BCE) uses
# logarithms to heavily punish confident wrong answers.
# If the label is 1.0 and the model predicts 0.9, the loss is small. 
# If the label is 1.0 and the model predicts 0.001, the log function
# makes the loss massive to signal a need to update weights and biases.
