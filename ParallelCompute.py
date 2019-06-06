"""
Given a set of initial conditions, propagate these initial conditions
in PARALLEL.
Input:
    v (3d array) set of initial conditions. Each v[:,:,i] is an initial condition in 2D.
    vt (3d array) set of initial conditions. Each v[:,:,i] is an initial condition in 2D
    vel (2d array) wavespeed field
    velX (2d array) wavespeed field at coarse scale 
    dx (double) spatial grid size
    dX (double) coarsened spatial grid size
    dt (double) temporal step size
    dT (double) coarsened temporal step size
    cT (double) terminal time
Output:
    ufx (3d array) solution of set of initial conditions at time cT, using fine config vel,dx,dt 
    utfx (3d array) solution of set of initial conditions at time cT, using fine config vel,dx,dt 
    UfX (3d array) coarsened solution of ufx
    UtfX (3d array) coarsened solution of utfx 
    UcX (3d array) solution of set of coarsened initial conditions at time cT, using coarse config velX,dX,dT 
    UtcX (3d array) solution of set of coarsened initial conditions at time cT, using coarse config velX,dX,dT
"""
import numpy as np
from skimage.transform import resize # for Coarsening 
import wave2

def ParallelCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):
    ncT = v.shape[2]
    ny,nx = velX.shape
    Ny,Nx = vel.shape
    
    # Allocate arrays for output
    ufx = np.zeros([Ny,Nx,ncT])
    utfx = np.zeros([Ny,Nx,ncT])
    UfX = np.zeros([ny,nx,ncT])
    UtfX = np.zeros([ny,nx,ncT])
    UcX = np.zeros([ny,nx,ncT])
    UtcX = np.zeros([ny,nx,ncT])
    
    # Parallel loop
    # Each rhs is independent of lhs
    for j in range(ncT-1):
            ufx[:,:,j+1],utfx[:,:,j+1] = wave2.wave2(v[:,:,j],vt[:,:,j],vel,dx,dt,cT)
            UfX[:,:,j+1] = resize(ufx[:,:,j+1],[ny,nx])
            UtfX[:,:,j+1] = resize(utfx[:,:,j+1],[ny,nx])
            UcX[:,:,j+1],UtcX[:,:,j+1] = wave2.wave2(resize(v[:,:,j],[ny,nx]),resize(vt[:,:,j],[ny,nx]),\
                                        velX,dX,dT,cT)
            
    return UcX,UtcX,UfX,UtfX,ufx,utfx
    
    