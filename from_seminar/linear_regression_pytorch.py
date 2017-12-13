import torch
from torch.autograd import Variable
from torch import nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
torch.manual_seed(3)


x_train = torch.Tensor([[1],[2],[3]])
y_train = torch.Tensor([[1],[2],[3]])

# define : y = W*x + b 
model = nn.Linear(1, 1, bias=True)
print (model)

cost_func = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""

x, y = Variable(x_train), Variable(y_train)

W = Variable(torch.rand(1,1))
#b = Variable(torch.rand(1,1))

cost_func = nn.MSELoss()

lr = 0.01

for step in range(300):
    prediction = x.mm(W)
    cost = cost_func(prediction, y)
    gradient = (prediction-y).view(-1).dot(x.view(-1)) / len(x)
    W -= lr * gradient

    if step %10 == 0:
        print(step, "going cost")
        print('[cost] :' ,cost)
        print('(prediction-y).view(-1) :', (prediction-y).view(-1))
        print('(x.view(-1) :', (x.view(-1)))
        print('gradient : ', gradient)
        print('W :', W)
"""
