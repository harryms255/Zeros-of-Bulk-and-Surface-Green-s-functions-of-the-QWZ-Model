# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:31:45 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *

t=1
alpha=0.25
mu=-0.5
kx_values=np.linspace(-np.pi,np.pi,501)
omega=0

bulk_spectrum=np.zeros((2,len(kx_values)))
surface_spectrum=np.zeros((2,len(kx_values)))

fig,axs=plt.subplots(1,2)

#GF spectrum
ax=axs[0]
for kx_indx,kx in enumerate(tqdm(kx_values)):
        
    bulk_spectrum[:,kx_indx]=np.linalg.eigvalsh(analytic_GF(omega, kx, 0, t, mu, alpha,eta=0.0001j))
    surface_spectrum[:,kx_indx]=np.linalg.eigvalsh(analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=0.0001j))
        

for j in range(2):
    if j==0:
        ax.plot(kx_values/np.pi,bulk_spectrum[j,:],"k-",label="Bulk")
        ax.plot(kx_values/np.pi,surface_spectrum[j,:],"b-",label="Surface")
    else:
        ax.plot(kx_values/np.pi,bulk_spectrum[j,:],"k-")
        ax.plot(kx_values/np.pi,surface_spectrum[j,:],"b-")
        
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\lambda_gt$")
ax.set_ylim(top=1,bottom=-1)
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.axhline(y=0,linestyle="dashed",linewidth=2.5,color="black")
ax.set_title("Eigen decomposition")


bulk_svd=np.zeros((2,len(kx_values)))
surface_svd=np.zeros((2,len(kx_values)))
#GF svd
ax=axs[1]
for kx_indx,kx in enumerate(tqdm(kx_values)):
        
    bulk_svd[:,kx_indx]=np.linalg.svd(analytic_GF(omega, kx, 0, t, mu, alpha,eta=0.0001j))[1]
    surface_svd[:,kx_indx]=np.linalg.svd(analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=0.0001j))[1]
        

for j in range(2):
    if j==0:
        ax.plot(kx_values/np.pi,bulk_svd[j,:],"k-",label="Bulk")
        ax.plot(kx_values/np.pi,surface_svd[j,:],"b-",label="Surface")
    else:
        ax.plot(kx_values/np.pi,bulk_svd[j,:],"k-")
        ax.plot(kx_values/np.pi,surface_svd[j,:],"b-")
        
ax.legend()
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\Sigma_gt$")
ax.set_ylim(top=1,bottom=-0.1)
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.axhline(y=0,linestyle="dashed",linewidth=2.5,color="black")
ax.set_title("SVD")