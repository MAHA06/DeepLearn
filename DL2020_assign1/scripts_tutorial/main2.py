### do more wrapping for the Model

import torch
import torch.nn as nn

## Data 

x = torch.Tensor([0.8345,0.0993,1.8054,1.8896,0.9817])
y = torch.Tensor([0.9785,0.6754,1.8001,0.7385,0.2224])

value = torch.Tensor([8.4596,2.2981,13.8385,11.3696,4.7279])

## Model

class Model(nn.Module):
    def __init__(self):

        super(Model, self).__init__()

        # define the parameters of the model here
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, data):

        # shape of the data is : 2 x 5

        f = self.a * data[0,:] + self.b * data[1,:]
        return f


## Training scheme

model = Model()
lr = 0.05

for i in range(1000):

    # define the forward propagation
    data = torch.stack([x, y])

    f = model(data)

    loss = torch.mean(torch.abs(value - f))

    if i % 100 == 1:
        print('a: %f | b: %f | Iteration: %d | Loss: %f' % (model.a.data, model.b.data, i, loss.data))

    # define the backward propagation

    model.zero_grad()

    loss.backward()
    model.a.data = model.a.data - lr * model.a.grad
    model.b.data = model.b.data - lr * model.b.grad

print('done')


