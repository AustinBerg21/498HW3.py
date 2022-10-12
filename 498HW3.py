import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

X1 = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
X2 = np.array([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0])
x1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
x2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)

P = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)

a = np.array(([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]))

T = 20

Pw = 10 ** (a[0, 0] - a[0, 1] / (T + a[0, 2]))
Pd = 10 ** (a[1, 0] - a[1, 1] / (T + a[1, 2]))
A = Variable(torch.tensor([2.0, 1.0]), requires_grad=True)          # initial guess, grad=true because we differentiate Pmodel with respect to A
print('Initial guess A12 and A21:',A)
alpha = 0.0001

for n in range(100):
    Pmodel = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * Pw + x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * Pd

    loss = (Pmodel - P)**2   # square error
    loss = loss.sum()

    loss.backward()             # computes gradient for current guess to get instant gradient

    with torch.no_grad():
        A -= alpha*A.grad         # gradient descent step

        A.grad.zero_()           # prevents gradient from being added everytime backward is called

print('Estimation A12 and A21:',A)
print('Final loss:',loss.data.numpy())

import matplotlib.pyplot as plt
Pmodel = Pmodel.detach().numpy()
P = P.detach().numpy()
x1 = x1.detach().numpy()

plt.plot(x1, Pmodel, label='Model Pressure')
plt.plot(x1, P, label='Exact Pressure')
plt.xlabel('x1')
plt.ylabel('Pressure')
plt.legend()
plt.title('Model vs. Exact Pressure')
plt.show()
