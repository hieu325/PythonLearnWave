"""
Solving 2D second order wave equation with periodic BC
Input: 
    u_0 (2d array) initial condition
    ut_0 (2d array) initial condition
    vel (2d array) wavespeed field
    dx (double) spatial grid 
    dt (double) temporal step size
    Tf (double) terminal time
Output:
    u (2d array) solution at Tf
    ut (2d array) solution at Tf
"""
import numpy as np

# Define periodic Laplacian
def periLaplacian2(v,dx):
    Lv = (np.roll(v,1,axis=1) - 2*v + np.roll(v,-1,axis=1))/dx + \
         (np.roll(v,1,axis=0) - 2*v + np.roll(v,-1,axis=0))/dx
    return Lv

# Wave solution propagator
def wave2(u0,ut0,vel,dx,dt,Tf):
    Nt = round(Tf/dt)
    c2 = np.multiply(vel,vel)
    
    u = u0
    ut = ut0
    
    for i in range(Nt):
        # Velocity Verlet
        ddxou = periLaplacian2(u,dx)
        u = u + dt*ut + 0.5*dt**2*np.multiply(c2,ddxou)
        ddxu = periLaplacian2(u,dx)
        ut = ut + 0.5*dt*np.multiply(c2,ddxou+ddxu)
    
    return u,ut