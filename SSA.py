# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:57:32 2021

@author: lysen
"""
import numpy as np
import pandas as pd

def ssa_trend(x, L, n_components, n_forcast = 0):
    # Step1 : Build trayectory matrix
    N = x.size
    if L > N / 2:
        L = N - L
    
    K = N - L + 1
    X = np.zeros((L,K))
    for i in range(K):
        X[:L,i] = x[i:i+L]
        
    # Step 2: SVD
    S = X.dot(X.transpose())
    autoval, U = np.linalg.eig(S)
    ranked = np.argsort(autoval)
    
    largest_indices = ranked[::-1][:n_components]
    U = U[::,largest_indices]
    V = (X.transpose()).dot(U)
    
    # Step 3: Grouping
    PC = U[::,:n_components]
    VT = V.transpose()
    rca = U[::,:n_components].dot(VT[:n_components,::])
    
    # Step 4: Reconstruction
    y = np.zeros(x.shape)
    Lp = np.min([L,K])
    Kp = np.max([L,K])
    for k in range(Lp-1):
        y[k] = 0
        for m in range(1,k+2):
            y[k] += (1.0/(k+1))*rca[m-1,k-m+1]
    
    for k in range(Lp-1,Kp):
        y[k] = 0
        for m in range(1,Lp+1):
            y[k] += (1.0/Lp)*rca[m-1,k-m+1]
            
    for k in range(Kp,N):
        y[k] = 0
        for m in range(k-Kp+2,N-Kp+2):
            y[k] += (1.0/(N-k))*rca[m-1,k-m+1]
    
    y = y.reshape(x.shape)
    
    if n_forcast > 0:
        u = PC[-1,::]
        Uc = PC[:-1,::]
        Uc_t = Uc.transpose()
        xc = x[-L+1:,np.newaxis]
    
    x_forcast = np.zeros((n_forcast,))
    for n in range(n_forcast):
        h = np.linalg.solve(Uc_t.dot(Uc), Uc_t.dot(xc))
        x_next = u.dot(h)
        x_forcast[n] = x_next;
        xc = np.vstack((xc[1:], x_next))

    return y, x_forcast