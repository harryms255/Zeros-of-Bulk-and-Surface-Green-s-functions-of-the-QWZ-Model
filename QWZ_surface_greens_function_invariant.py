# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:58:56 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *

def surface_winding_number(t,mu,alpha,eta=1e-06j):
    U=np.array(([1,1],[1j,-1j]))
    kx_values=np.linspace(-np.pi,np.pi,101)
    z_values=np.zeros(len(kx_values),dtype=complex)
    z_grad_values=np.zeros(len(kx_values),dtype=complex)
    dk=kx_values[1]-kx_values[0]
    for kx_indx,kx in enumerate(kx_values):
        H=np.linalg.inv(TB_surface_Greens_function(0, kx, 101, t, mu, alpha))
        
        z_values[kx_indx]=(np.conj(U.T)@H@U)[0,1]
    
    for i in range(len(z_values)):
        z_grad_values[i]=(z_values[i]-z_values[i%len(kx_values)])/dk
        
    winding_number=1/(2*np.pi*1j)*sum(z_grad_values/z_values)*dk
    
    return np.real(winding_number)
        
        


t=1
mu_values=np.linspace(-4,4,101)
alpha=0.25

# invariant_values=np.zeros((len(mu_values)))

# for mu_indx,mu in enumerate(tqdm(mu_values)):
#     invariant_values[mu_indx]=surface_winding_number(t, mu, alpha)
    
# plt.figure()
# plt.plot(mu_values,invariant_values,"k-")


# mu_values=np.linspace(-4,4,21)
kx_values=np.linspace(-np.pi,np.pi,501)
# spectrum=np.zeros((2,len(kx_values)))

# for mu in mu_values:
#     for kx_indx,kx in enumerate(tqdm(kx_values)):
#         spectrum[:,kx_indx]=np.linalg.eigvalsh(np.linalg.inv(analytic_GF(0, kx, 0, t, mu, alpha,eta=0.000001j)))

        
#     plt.figure()
#     plt.plot(kx_values,spectrum[0,:],"k-")
#     plt.plot(kx_values,spectrum[1,:],"k-")
#     plt.title(r"$\mu={:.1f}t$".format(mu))


mu=0.5
omega_values=np.linspace(-1,1,501)
det_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx,kx in enumerate(kx_values):
        det_values[omega_indx,kx_indx]=abs(np.linalg.det(np.linalg.inv(analytic_GF(omega, kx, 0, t, mu, alpha))))
        
plt.figure()
sns.heatmap(det_values,cmap="plasma",vmax=10)
plt.gca().invert_yaxis()

