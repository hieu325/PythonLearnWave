"""
Created on Sat May 18 01:47:27 2019

@author: huuhi
"""

import time
import numpy as np
import wave2
#import parareal2
#import matplotlib.pyplot as plt
import ParallelCompute

def initCond(xx,yy):
    u0 = np.exp(-(xx**2 + 0*yy**2))
    ut0 = np.zeros([np.size(xx,axis=1),np.size(yy,axis=0)])
    return u0,ut0

def wavespeed(xx,yy):
    vel = 1+0*xx
    return vel

if __name__ == "__main__": 

    # Parameter setups
    T = 2
    cT = 0.05
    ncT = round(T/cT)
    dx = 0.01
    dt = dx/50
    x = np.arange(-1,1,dx)
    y = np.arange(-1,1,dx)
    xx,yy = np.meshgrid(x,y)
    vel = wavespeed(xx,yy)
    u0,ut0 = initCond(xx,yy)
    
    uf = np.zeros([np.size(u0,axis=0),np.size(u0,axis=1),ncT])
    utf = np.zeros([np.size(u0,axis=0),np.size(u0,axis=1),ncT])
    
    uf[:,:,0] = u0
    utf[:,:,0] = ut0
    
    # Compute serial fine solution as reference
    startp = time.time()
    for i in range(ncT-1):
        uf[:,:,i+1],utf[:,:,i+1] = wave2.wave2(uf[:,:,i],utf[:,:,i],vel,dx,dt,cT)
    endp = time.time()
    print('fine serial time ',endp-startp)
     
    # Coarsening config
    dX = 0.05
    dT = dX/10
    X = np.arange(-1,1,dX)
    Y = np.arange(-1,1,dX)
    XX,YY = np.meshgrid(X,Y)
    velX = wavespeed(XX,YY)
    
    # Test Parallel Compute function
    startp = time.time()
    UcX,UtcX,UfX,UtfX,ufx,utfx = ParallelCompute.ParallelCompute(uf,utf,vel,velX,dx,dX,dt,dT,cT)
    endp = time.time()
    print('Parallel compute time ',endp-startp)
    print('If parallel loop is effective, parallel computation time should be smaller than the serial time')
    
    
    # Test parareal method
    #m = 1
    #tm = 5
    #pimax = 5
    #startp = time.time()
    #up,utp = parareal2.parareal2_orth(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax)
    #endp = time.time()
    #print('parareal time ',endp-startp)
    
    # Plot solution
    
    
    # Plot error
    #relUerror = np.zeros([ncT,pimax])
    #for i in range(ncT):
    #    for pI in range(pimax):
    #        relUerror[i,pI] = np.sqrt(np.sum(np.abs(up[:,:,i,pI]-uf[:,:,i])**2))
    
    #plt.semilogy(relUerror)
    
    
    
    
