# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:10:07 2025

@author: Harry MullineauxSanders
"""

import numpy as np
import qsymm
import sympy

QWZ_model="""
   a*cos(kx)*sigma_x+a*cos(ky)*sigma_y+(mu-t*(sin(kx)+sin(ky)))*sigma_z
  """

QWZ_model_qsymm=qsymm.Model(QWZ_model,momenta=["kx","ky"])
discrete_symm, continuous_symm = qsymm.symmetries(QWZ_model_qsymm,qsymm.groups.square())

for i in range(len(discrete_symm)):
    display(discrete_symm[i])
    print(np.round(discrete_symm[i].U,decimals=10))
    print("Conjugate={}".format(discrete_symm[i].conjugate))