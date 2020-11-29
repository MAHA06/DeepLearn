### very naive implementation

import torch

## Data 

x = torch.Tensor([0.8345,0.0993,1.8054,1.8896,0.9817])
y = torch.Tensor([0.9785,0.6754,1.8001,0.7385,0.2224])

value = torch.Tensor([8.4596,2.2981,13.8385,11.3696,4.7279])


## Model

a = torch.rand(1)
b = torch.rand(1)

a.requires_grad = True
b.requires_grad = True


## Training scheme

# iteration set the number of iteration to 1000
lr = 0.01

for i in range(1000):

    # calculate the loss, forward propagation

    loss = torch.mean(torch.abs(value - (a * x + b * y)))

    if i % 100 == 1:
        print('a: %f | b: %f | Iteration: %d | Loss: %f' % (a.data, b.data, i, loss.data))

    # backward propagation

    if a.grad is not None:
        a.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    loss.backward()

    a.data = a.data - lr * a.grad
    b.data = b.data - lr * b.grad


print('done')


