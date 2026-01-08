# A.4 Automatic Differentiation Made Easy

# If we carry out computations in PyTorch, it will build a computational graph internally by default 
# if one of its terminal nodes has the requires_grad attribute set to True. This is useful if we want to 
# compute gradients. Gradients are required when training neural networks via the popular backpropagation 
# algorithm, which can be considered an implementation of the chain rule from calculus for neural networks.

# The most common way of computing the loss gradients in a computation graph involves applying the chain rule 
# from right to left, also called reverse-model automatic differentiation or backpropagation. We start from the output 
# layer (or the loss itself) and work backward through the network to the input layer. We do this to compute the 
# gradient of the loss with respect to each parameter (weights and biases) in the network, which informs how we 
# update these parameters during training.

# Partial derivatives measure the rate at which a function changes with respect to one of its variables. A gradient is a 
# vector containing all of the partial derivatives of a multivariate function, a function with more than one variable as input.

# The chain rule is a way to compute gradients of a loss function given the model’s parameters in a computation graph. 
# This provides the information needed to update each parameter to minimize the loss function, which serves as a proxy 
# for measuring the model’s performance using a method such as gradient descent. 
