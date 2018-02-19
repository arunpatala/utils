import torch as th
import matplotlib.pyplot as plt
import numpy as np

def random(N): return th.rand(N)
def pm(N): return (random(N)<0.5).float()*2 - 1
def betap(N, alpha=0.2): return  th.from_numpy(np.random.beta(alpha, alpha+1, N)).float()
def betan(N, alpha=0.2): return  -betap(N, alpha) 
def betapn(N, alpha=0.2): return  betap(N, alpha) * pm(N)
def unifp(N, alpha=0.2): return (random(N)<alpha).float() * random(N)
def unifn(N, alpha=0.2): return -unifp(N, alpha)
def unifpn(N, alpha=0.2): return unifp(N, alpha) * pm(N)
def binp(N, alpha=0.2): return (random(N)<alpha).float()
def binn(N, alpha=0.2): return -1 * binp(N, alpha)
def binpn(N, alpha=0.2): return binp(N, alpha) * pm(N)


def mixup(X, alpha, fun, X2=0, mode=None, shuffle=False):
    s = X.shape
    if shuffle: X2 = X[th.randperm(len(X))]
    if mode is not None:
        if mode=='channel': s = s[:1] + tuple([1]*(len(s)-2)) + s[-1:]
        else: s = s[:mode] + tuple([1]*(len(s)-mode))
    #print("mixup shape", s)
    P = fun(s, alpha)
    X12 = (X-X2)
    X12 = Variable(P, requires_grad=False) * X12
    return X + X12

def shake_shake(X1, X2, mode=None):
    return mixup(X1, 1.0, unifn, X2, mode=mode)
def shake_shake_unbiased(X1, X2, mode=None):
    return mixup(X1, 1.0, unifpn, X2, mode=mode)
def shake_1branch(X, alpha, mode=None):
    return mixup(X, alpha, unifp,  mode=mode)
def shake_1branch_unbiased(X, alpha, mode=None):
    return mixup(X, alpha, unifpn,  mode=mode)

def stochastic_drop(X, alpha, mode=0):
    return mixup(X, alpha, binn,  mode=mode)  
def stochastic_drop_unbiased(X, alpha, mode=0):
    return mixup(X, alpha, binpn,  mode=mode)    
def shake_drop(X, alpha, mode=0):
    return mixup(X, alpha, unifn,  mode=mode)  
def shake_drop_unbiased(X, alpha, mode=0): 
    return mixup(X, alpha, unifpn,  mode=mode) 

def mixup_layer(X1, alpha,  mode=1, shuffle=True):
    return mixup(X1, alpha, betan,  mode=mode)
def mixup_layer_unbiased(X1, alpha, mode=1, shuffle=True):
    return mixup(X1, alpha, betapn,  mode=mode, shuffle=shuffle)

def dropout(X1, alpha, mode='channel'):
    return mixup(X1, alpha, binn, mode=mode)
def dropout_unbiased(X1, alpha, mode='channel'):
    return mixup(X1, alpha, binpn, mode=mode)
def dropout_unif(X1, alpha, mode='channel'):
    return mixup(X1, alpha, unifpn, mode=mode)
def dropout_beta(X1, alpha, mode='channel'):
    return mixup(X1, alpha, betapn, mode=mode)