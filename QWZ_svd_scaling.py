# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:42:57 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *
from scipy.optimize import curve_fit

def linear_fit(x,a,b,c):
    return a*(x-b)+c

Ny=51
t=1
alpha=0.25
mu=1
SB=0.25
kx_values=np.linspace(-0.4*np.pi,0.4*np.pi,501)
omega_values=np.linspace(-1,1,251)
zero_values=np.zeros((len(kx_values)))
SB_zero_values=np.zeros((len(kx_values)))
pole_values=np.zeros(len(kx_values))


kx_values=np.linspace(0,0.4*np.pi,21)
omega_values=np.linspace(0,0.3,501)

for kx in tqdm(kx_values):
    fig,ax=plt.subplots(num="kx={:.2f}pi".format(kx/np.pi))
    zero=analytic_luttinger_surface(kx, t, mu, alpha)[0]
    
    SVD_values=np.zeros((2,len(omega_values)))
    
    for omega_indx,omega in enumerate(omega_values):
        SVD_values[:,omega_indx]=np.linalg.svd(analytic_surface_Greens_function(omega, kx, t, mu, alpha))[1]
    
    
    log_omega=np.log10(abs(omega_values-zero))
    log_omega_sort=np.argsort(log_omega)
    
    log_omega_sorted=(log_omega[log_omega_sort])[1:50]
    log_SVD=(np.log10(SVD_values)[:,log_omega_sort])[:,1:50]
    
    zero_fit,zero_err=curve_fit(linear_fit,log_omega_sorted,log_SVD[1,:])
    pole_fit,pole_err=curve_fit(linear_fit,log_omega_sorted,log_SVD[0,:])
    
    
    
    
    
    
    
    
    
    
    colors=["r","g"]
    label=["Pole, gradient={:.3f}".format(pole_fit[0]),"Zero, gradient={:.3f}".format(zero_fit[0]),]
    for i in range(2):
        ax.plot(log_omega_sorted,log_SVD[i,:],"{}".format(colors[i]),label="{}".format(label[i]),linewidth=4)
    ax.set_xlabel(r"$\log_{10}((\omega-P)/t)$")
    ax.set_ylabel(r"$\log_{10}(\Sigma_{g(\omega,k_x,y=0)})$")
    #ax.set_ylim(bottom=0,top=10)
    ax.set_title(r"$k_x={:.2f}\pi$".format(kx/np.pi))
    plt.legend()
    plt.plot(log_omega_sorted,linear_fit(log_omega_sorted, *zero_fit),"k",linestyle="dashed",linewidth=3)
    plt.plot(log_omega_sorted,linear_fit(log_omega_sorted, *pole_fit),"k",linestyle="dashed",linewidth=3)