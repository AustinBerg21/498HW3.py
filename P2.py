import numpy
from bayes_opt import BayesianOptimization

def f(x1, x2):
    return -1*(((4-2.1 * (x1) **2 + (x1 ** 4    ) / 3) * x1**2 + x1*x2 + (-4 + 4 * x2 **2) * x2 **2))

pbounds = {'x1': (-3, 3), 'x2': (-2, 2)}
optimizer = BayesianOptimization(f=f, pbounds=pbounds, random_state=1)

optimizer.maximize(init_points=1203, n_iter=1203,)
## After doing 2000 points found the minimimum of the function to be -1.02819.
# Using the bayesian optimization toolbox I was able to use the maximize operation of the negative of the function.
# This in return would give the x1 and x2 values that would minimize the actual function. All the function targets in the 1st column are the negative of the actual function targets in the problem.
# This is how I got 1.02819 with the x1 = .1158 and x2 = -.7243.
# Afterwards I plugged in these values in the original equation and got -1.02819
# The off the shelf method I used was from https://github.com/fmfn/BayesianOptimization

