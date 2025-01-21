# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:34:08 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *
Ny=51
t=1
alpha=0.25
mu=1
kx_values=np.linspace(-0.46*np.pi,0.46*np.pi,501)
omega_values=np.linspace(-1,1,251)
zero_values=np.zeros((2,len(kx_values)))

fig,axs=plt.subplots(1,2)

ax=axs[0]
TB_spectrum=np.zeros((2*Ny,len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    TB_spectrum[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha))
for i in range(2*Ny):
    if i==0:
        ax.plot(kx_values/np.pi,TB_spectrum[i,:],"k-",label="Tight Binding",linewidth=3)
    else:
        ax.plot(kx_values/np.pi,TB_spectrum[i,:],"k-",linewidth=3)



for kx_indx,kx in enumerate(tqdm(kx_values)):
    zero_values[0,kx_indx],zero_values[1,kx_indx]=analytic_bulk_luttinger_surface(kx, t, mu, alpha,eta=0.001j)
ax.plot(kx_values/np.pi,zero_values[0,:],"r--",label="Zeros",linewidth=3)
ax.plot(kx_values/np.pi,zero_values[1,:],"r--",linewidth=3)
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.set_ylim(top=1.5,bottom=-1.5)
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\omega/t$")
ax.legend()

ax=axs[1]
kx=0.2*np.pi
omega_values=np.linspace(-0.3,0.3,501)

SVD_values=np.zeros((2,len(omega_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    SVD_values[:,omega_indx]=np.linalg.svd(analytic_GF(omega, kx, 0, t, mu, alpha))[1]

for i in range(2):
    ax.plot(omega_values,SVD_values[i,:],"k-")
ax.set_xlabel(r"$\omega/t$")
ax.set_ylabel(r"$\Sigma_{g(\omega,k_x,y=0)}$")
ax.set_ylim(bottom=0,top=10)
ax.set_title(r"$k_x={:.2f}\pi$".format(kx/np.pi))
#ax.axhline(y=0,linewidth=3,linestyle="dashed")
ax.set_xlim(left=min(omega_values),right=max(omega_values))

