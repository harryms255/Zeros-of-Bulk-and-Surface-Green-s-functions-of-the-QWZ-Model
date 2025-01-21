# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:52:38 2025

@author: Harry MullineauxSanders
"""


from QWZ_functions_file import *


t=1
alpha=0.25
mu=1
kx_values=np.linspace(-np.pi,np.pi,501)
omega_values=np.linspace(-0.3,0.3,501)

det_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx,kx in enumerate(kx_values):
        det_values[omega_indx,kx_indx]=abs(np.linalg.det(analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=0.000001j)))
        
plt.figure()
sns.heatmap(det_values,cmap="plasma")
plt.gca().invert_yaxis()