"""
Python Source File Header

This file contains a collection of functions and utilities designed for 
DL-SCA mathematical modeling, gradient computation, and optimization within 
a specific framework detailed in section 2 of the paper: 
Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking.

The code primarily utilizes symbolic and numerical 
computation libraries to define, manipulate, and evaluate loss functions 
and their derivatives.

Non-Exhaustive List of Dependencies:
- sympy: For symbolic mathematics, including differentiation and simplification.
- joblib: For parallelization and task execution. By default, the code uses
    all available cores for parallel computation. Set n_jobs to a specific
    number to limit the number of cores used.

Key Functionalities:
1. **Mathematical Models**: Define and combine single-parameter models into more 
   complex schemes.
2. **Gradient and Loss Calculations**: Compute gradients, loss values, and their 
   symbolic derivatives for optimization.
3. **Hessian and Estimators**: Evaluate and manipulate the Hessian matrix for 
   advanced optimization techniques.
4. **Loss Landscape Analysis**: Compute horizontal and vertical gradients and 
   gradient norms of the loss landscape.
5. **Projection and Mapping**: Perform parameter projection onto dual spaces

If you are using this code, please cite the aforementioned paper in your work.

Author: Nathan Rousselot
Date: [2025-01-13]
Version: 1.0
"""


import numpy as np
from sympy import symbols, log, diff, exp, print_latex, simplify, Abs, sqrt
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def single_model(theta, L):
    '''
    Single parameter model $\mathbf{F}_i(\alpha_is_i|\theta_i)$ as per Eq. 2.3.2
    '''
    M0 = theta*L
    f = [M0, 1-M0]
    return f

def combine_any_models(f):
    '''
    Combining $n+1$ single parameter model into the so-called scheme-aware model.
    Based on Eq. 2.3.3
    The code below uses Poisson equation to compute the convolution and presupposes
    boolean masking scheme.
    '''
    n_bits = len(f)
    M0 = 0
    M1 = 0
    for i in range(2**n_bits):
        i_bits = [int(x) for x in list(bin(i)[2:].zfill(n_bits))]
        if sum(i_bits) % 2 == 0:
            M0 += np.prod([f[j][i_bits[j]] for j in range(n_bits)])
        else:
            M1 += np.prod([f[j][i_bits[j]] for j in range(n_bits)])
    return [simplify(M0), simplify(M1)]

def softmax(f):
    '''
    Compute the arg-softmax function for a $n=1$ scheme-aware model.
    To generalize on any $n$, simply loop as for \texttt{combine\_any\_models(f)}.
    '''
    return [simplify(exp(f[0])/(exp(f[0])+exp(f[1]))), simplify(exp(f[1])/(exp(f[0])+exp(f[1])))]    

def log_outputs(f):
    for i in range(len(f)):
        f[i] = simplify(log(f[i]))
    return f

def NLL(f, s):
    '''
    Computes the negative log-likelihood of the model given the secret $s$.
    '''
    if s == 0:
        return -f[0]/np.log(2)
    if s == 1:
        return -f[1]/np.log(2)

def compute_gradients(loss, theta0, theta1):
    '''
    Compute the gradient of the loss function with respect to the parameters $\theta_0$ and $\theta_1$.
    '''
    loss_g = [diff(loss, theta0), diff(loss, theta1)]
    return loss_g

def calculate_loss(theta0x, theta1y, theta0, theta1, cumulated_loss):
    '''
    Evaluate the loss function given the parameters $\theta_0$ and $\theta_1$.
    '''
    return cumulated_loss.subs(theta0, theta0x).subs(theta1, theta1y).evalf()

def l1_eps_mirror_map(x, eps=0.1):
    return (1+eps)*(np.abs(x)**eps)*np.sign(x)

def project_axis(thetas, eps=0.1):
    '''
    Project the parameters $\theta_0$ and $\theta_1$ onto the $\ell_{1+\epsilon}$ ball.
    '''
    thetas_dual = thetas
    for i in range(len(thetas)):
        thetas_dual[i] = l1_eps_mirror_map(thetas[i], eps=eps)
    return thetas_dual

def compute_hessian(loss, theta0, theta1):
    '''
    Compute the Hessian of the loss function with respect to the parameters $\theta_0$ and $\theta_1$.
    '''
    loss_h = [[diff(diff(loss, theta0), theta0), diff(diff(loss, theta0), theta1)],
              [diff(diff(loss, theta1), theta0), diff(diff(loss, theta1), theta1)]]
    return loss_h

def hutchinson(hessian):
    '''
    Compute the Hutchinson estimator of the Hessian.
    '''
    z = np.random.choice([-1,1], 2)
    hessian_vp = hessian@z
    return hessian_vp*z

def clip_diag(hessian, min=1e-7, max = None):
    if max is None and min is not None:
        return np.clip(hessian, min, None)
    elif min is None and max is not None:
        return np.clip(hessian, None, max)
    elif min is not None and max is not None:
        return np.clip(hessian, min, max)
    else:
        return hessian

def list_of_list_to_np(hessian):
    hessian_np = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            hessian_np[i,j] = hessian[i][j]
    return hessian_np

def evaluate_hessian(hessian, theta0x, theta1y, theta0, theta1, epsilon=1e-7):
    '''
    Evaluates the Hessian at a given point $(\theta_0, \theta_1)$.
    '''
    for i in range(2):
        for j in range(2):
            value = hessian[i][j].subs(theta0, theta0x).subs(theta1, theta1y).evalf()
            value += epsilon
            hessian[i][j] = value
    return hessian

def h_grad(loss_landscape):
    '''
    Compute the horizontal gradient of the loss landscape.
    '''
    h_grad = np.zeros_like(loss_landscape)
    for i in range(np.shape(loss_landscape)[0]):
        for j in range(np.shape(loss_landscape)[1]):
            if i == 0:
                h_grad[i,j] = loss_landscape[i+1,j]-loss_landscape[i,j]
            elif i == np.shape(loss_landscape)[0]-1:
                h_grad[i,j] = loss_landscape[i,j]-loss_landscape[i-1,j]
            else:
                h_grad[i,j] = (loss_landscape[i+1,j]-loss_landscape[i-1,j])/2
    return h_grad

def v_grad(loss_landscape):
    '''
    Compute the vertical gradient of the loss landscape.
    '''
    v_grad = np.zeros_like(loss_landscape)
    for i in range(np.shape(loss_landscape)[0]):
        for j in range(np.shape(loss_landscape)[1]):
            if j == 0:
                v_grad[i,j] = loss_landscape[i,j+1]-loss_landscape[i,j]
            elif j == np.shape(loss_landscape)[1]-1:
                v_grad[i,j] = loss_landscape[i,j]-loss_landscape[i,j-1]
            else:
                v_grad[i,j] = (loss_landscape[i,j+1]-loss_landscape[i,j-1])/2
    return v_grad

def grad_norm(h_grad, v_grad):
    '''
    Compute the norm of the gradient of the loss landscape.
    '''
    return np.sqrt(h_grad**2+v_grad**2)

def get_gradient_slice(ll_grad, thetax):
    '''
    Takes a slice of the gradient landscape at a given $\theta_x$.
    '''
    n,m = np.shape(ll_grad)
    grad_slice = []
    i = 0
    while i < m:
        grad_slice.append(ll_grad[thetax,i])
        i += 1
    return grad_slice