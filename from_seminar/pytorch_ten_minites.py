import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package

# 1st examples
"""
#d = [ [[1., 2.,3.], [4.,5.,6.]], [[7.,8.,9.], [11.,12.,13.]] ]
d = [ [[1., 28.,34.], [2.,5.,9.]], [[10.,3.,5.], [2.,2.,1.]] ]
d = torch.Tensor(d)
#print ("shape of the tensor:", d.size())

#z = d[0] + d[1]
#print ("adding up the two matrices of the 3d tensor:", z)

print (d.view(2, -1))

x = autograd.Variable(d, requires_grad=True)
print ("the node's data is the tensor:", x.data.size())
print ("the node's gradient is empty at creation:", x.grad)

y = x + 1
print ("x and y:", x, y)
z = x + y
print ("z :", z)
s = z.sum()
print (s.creator)
print ("s :", s)

s.backward()
print ("the variable now has gradients:", x.grad)
"""

# 2nd examples : linear transformation of a 2x5 matrix into 2x3 matrix
"""
linear_map = nn.Linear(5, 3)
print ("using randomly initialized params:", linear_map.parameters)

data = torch.randn(2,5)
y = autograd.Variable(torch.randn(2,3))

# make a node
x = autograd.Variable(data, requires_grad=True)

a = linear_map(x)
z = F.relu(a)
o = F.softmax(z)

print ("output of softmax as a probability distribution:", o.data.view(1, -1))

# loss function
loss_func = nn.MSELoss()
L = loss_func(z, y)
print ("Loss:", L)

class Log_reg_classifier(nn.Module):
    def __init__(self, in_size, out_size):
        super(Log_reg_classifier, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, vect):
        return F.log_softmax(self.linear(vect))

linear_map = nn.Linear(5, 3)
optimizer = optim.SGD(linear_map.parameters(), lr = 1e-2)

# epoch loop: we run following until convergence
optimizer.zero_grad() # make gradient zero
L.backward(retain_variables = True)
optimizer.step()
print (L)
"""
class Log_reg_classifier(nn.Module):
    def __init__(self, in_size, out_size):
        super(Log_reg_classifier, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, vect):
        return F.log_softmax(self.linear(vect))

# 3rd examples : Building a Neural Network
data = torch.randn(2,5)
# i) define model
model = Log_reg_classifier(10, 2)

# ii) define loss function
loss_func = nn.MSELoss()

# iii) define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-1)

# iv) send data through model in minibatches for 10 epoche
for epoch in range(10):
    for minibatch, target in data:
        model.zero_grad()

        # forward pass
        out = model(autograd.Variable(minibatch))

        # backward pass
        L = loss_func(out, target)  # calculate loss
        L.backward()                # calculate gradients
        optimizer.step()            # make an update step


print (L)
