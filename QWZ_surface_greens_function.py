# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:22:58 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import itertools as itr
# plt.close("all")


# def QHZ_real_space(kx,Ny,t,mu,alpha):
    
#     H=np.zeros((2*Ny,2*Ny))
    
#     for y in range(Ny-1):
#         H[2*y,2*(y+1)]=-t/2
#         H[2*y+1,2*(y+1)+1]=t/2
        
#         H[2*(y+1),2*y+1]=alpha
#         H[2*y,2*(y+1)+1]=-alpha
        
#     for y in range(Ny):
#         H[2*y,2*y+1]=alpha*np.sin(kx)
        
    
#     H+=np.conj(H.T)
    
    
#     for y in range(Ny):
#         H[2*y,2*y]=mu-t*np.cos(kx)
#         H[2*y+1,2*y+1]=t*np.cos(kx)-mu
        
        
#     return H

# def surface_Greens_function(omega,kx,Ny,t,mu,alpha):
#     G=np.linalg.inv((omega+0.0000001j)*np.identity(2*Ny)-QHZ_real_space(kx, Ny, t, mu, alpha))
    
#     GS=G[:2,:2]
    
#     return GS

# def SDOS(omega,kx,Ny,t,mu,alpha):
#     #Surface Density of States
#     GS=surface_Greens_function(omega, kx, Ny, t, mu, alpha)
    
#     LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
#     return LDOS_values

Ny=251
t=1
alpha=0.1
mu=-1
kx_values=np.linspace(-np.pi,np.pi,501)

spectrum=np.zeros((2*Ny,len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha))

plt.figure()
for i in range(2*Ny):
    plt.plot(kx_values,spectrum[i,:],"k-") 
plt.xlabel(r"$k_x/\pi$")
plt.ylabel(r"$E/t$")
    
    
# omega_values=np.linspace(-np.pi,np.pi,501)

# LDOS_values=np.zeros((len(omega_values),len(kx_values)))

# for kx_indx,kx in enumerate(tqdm(kx_values)):
#     for omega_indx,omega in enumerate(omega_values):
#         LDOS_values[omega_indx,kx_indx]=SDOS(omega, kx, Ny, t, mu, alpha)
        
        
# plt.figure()
# sns.heatmap(LDOS_values,cmap="plasma",vmax=0.5)


#Greens function spectrum:
    
    
GF_spectrum=np.zeros((2,len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    GF_S=TB_surface_Greens_function(0, kx, Ny, t, mu, alpha)
    GF_S_herm=(GF_S+np.conj(GF_S.T))/2
    GF_spectrum[:,kx_indx]=np.linalg.eigvalsh(GF_S_herm)
    
    
plt.figure()
for i in range(2):
    plt.plot(kx_values,GF_spectrum[i,:],"k-")
plt.xlabel(r"$k_x/\pi$")
plt.ylabel(r"$\lambda t$")