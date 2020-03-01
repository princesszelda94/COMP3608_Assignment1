import torch as th
import torch.optim as optim
import torch.nn as nn
import numpy as np

d_array = np.load('d.npy')
E_array = np.load('E.npy')

#Convert into pytorch tensors
d = th.tensor(d_array, dtype=th.float32)
E = th.Tensor(list(E_array))
#E = th.from_numpy(E_array)

# d = th.tensor(np.load('d.npy'), dtype=th.float32)
# E = th.tensor(np.load('E.npy'), dtype=th.float32)


def loss_function(E, x, d):
    return th.norm((E @ x)-d)**2


class LeastSquaresContainer(nn.Module):
    def __init__(self, n):
        super().__init__()
        #x = th.from_numpy(np.random.random(n))
        x = th.tensor(np.random.random(n), dtype=th.float32)
        #print('x: ', x)
        self.x = nn.Parameter(x)

    def loss(self, E, d):
        return loss_function(E, self.x, d)


def least_squares_approx(E, d, lr=0.01, epochs=200):
    m, n = E.shape #[5000, 20] d is 5000
    estimator = LeastSquaresContainer(n=20) #n is the number of items in each array
    optimizer = optim.SGD(estimator.parameters(), lr=0.00001)
    num_epochs = 100
    # Training Loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # zero out gradients to prevent gradients from carrying over into other iterations
        loss = estimator.loss(E, d)  # compute loss
        #print('loss', loss)
        loss.backward()  # compute gradient with respect to loss function
        optimizer.step()  # use gradient descent to adjust value of parameters in model
    return estimator

estimator = least_squares_approx(E, d)
estimator.x 
print(estimator.x)

