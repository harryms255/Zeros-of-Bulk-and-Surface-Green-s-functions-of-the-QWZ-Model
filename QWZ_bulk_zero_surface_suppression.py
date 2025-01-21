# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:12:00 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *

def GH(omega,kx,t,mu,alpha,eta=0.0001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha],[alpha,t/2]))
    G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
    return (H0@G0+H1@G1)

def GHG(omega,kx,t,mu,alpha,eta=0.0001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha],[alpha,t/2]))
    G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
    return (H0@G0+H1@G1)@np.linalg.inv(G0)



t=1
alpha=0.25
mu=1
kx=0.2*np.pi
omega_values=np.linspace(-0.3,0.3,1001)


fig,axs=plt.subplots(2,2)

SVD_values=np.zeros((8,len(omega_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    SVD_values[:2,omega_indx]=np.linalg.svd(analytic_GF(omega, kx, 0, t, mu, alpha))[1]
    SVD_values[2:4,omega_indx]=np.linalg.svd(GH(omega, kx, t, mu, alpha))[1]
    SVD_values[4:6,omega_indx]=np.linalg.svd(GHG(omega, kx, t, mu, alpha))[1]
    SVD_values[6:8,omega_indx]=np.linalg.svd(analytic_surface_Greens_function(omega, kx, t, mu, alpha))[1]
    

for i in range(4):
    ax=axs[int(i/2),i%2]
    for j in range(2):
        ax.plot(omega_values,SVD_values[2*i+j,:],"k-")
    ax.set_xlabel(r"$\omega/t$")
    ax.set_ylabel(r"$\Sigma/t$")
    ax.set_ylim(bottom=0,top=5)
    ax.set_xlim(left=min(omega_values),right=max(omega_values))
    
axs[0,0].set_title(r"$g(\omega,0.2\pi,0)$")
axs[0,1].set_title(r"$H_0g(\omega,0.2\pi,0)\\+H_1g(\omega,0.2\pi,1)$")
axs[1,0].set_title(r"$[H_0g(\omega,0.2\pi,0)\\+H_1g(\omega,0.2\pi,1)]g^{-1}(\omega,0.2\pi,0)$")
axs[1,1].set_title(r"$g_S(\omega,0.2\pi)$")
