# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:55:09 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *


Ny=101
t=1
alpha=1
mu=-1
kx_values=np.linspace(-np.pi,np.pi,101)

spectrum_recursive=np.zeros((2,len(kx_values)))
spectrum_exact=np.zeros((2,len(kx_values)))
omega=0

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum_recursive[:,kx_indx]=np.linalg.eigvalsh(recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha))
    spectrum_exact[:,kx_indx]=np.linalg.eigvalsh(TB_surface_Greens_function(omega, kx, Ny, t, mu, alpha))

plt.figure()
for i in range(2):
    if i==0:
        plt.plot(kx_values, spectrum_recursive[i,:],"k-",label="Recursive")
        plt.plot(kx_values, spectrum_exact[i,:],"b.",label="Exact")
    else:
        plt.plot(kx_values, spectrum_recursive[i,:],"k-")
        plt.plot(kx_values, spectrum_exact[i,:],"b.")
        
#plt.ylim(top=1,bottom=-1)
plt.legend()




