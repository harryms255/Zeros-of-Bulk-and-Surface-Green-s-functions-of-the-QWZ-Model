# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:43:42 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *


t=1
alpha=0.25
mu=-0.5
kx_values=np.linspace(-np.pi,np.pi,101)

N1_values=np.zeros(len(kx_values),dtype=complex)

for kx_indx,kx in enumerate(tqdm(kx_values)):
    N1_values[kx_indx]=N1_local_invariant(kx, t, mu, alpha)
    
plt.figure()
plt.plot(kx_values,N1_values)

