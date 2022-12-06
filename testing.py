import numpy as np
import matplotlib.pyplot as plt
import json
import os
plt.rcParams["figure.figsize"] = (25,10)

plt.rcParams.update({'font.size': 22})

colors='bgrykcm'
symbol='*-+^'

from utils import forward_backwards_integration_and_merge_value

from Differential_Problem_Quan_Harm_Osci import DP_QHO as diff_prob
from Integrators import predictor_corrector_RK4_Adams_Moulton4 as integrator

E_grid= np.linspace(0.3, 0.6, num=37)
grid= np.linspace(-3.5, 3.5, num=3000) + 0.001
grid_b= np.flip(grid)

u0=((1.0/np.pi)**(1.0/4.0))*np.exp((-1.0*grid**2.0)/2.0)
u0_b=((1.0/np.pi)**(1.0/4.0))*np.exp((-1.0*grid_b**2.0)/2.0)
E0=0.5
u1=(2.0**0.5)*np.multiply(grid,u0)
E1=1.5

delta= abs(grid[1] - grid[0])
delta_b= abs(grid[-1] - grid[-2])

y1=u0[0]
y2=((u0[1]-u0[0])/delta)
y1_b=u0_b[0]
y2_b=((u0_b[1]-u0_b[0])/delta)

#energy search
merge_value_arra=[]
for E in E_grid:
    param_f={'grid':grid,
             'y_arra':list([[y1],[y2]]),
             'E':E}
    param_b={'grid':np.flip(grid),
             'y_arra':list([[y1_b],[y2_b]]),
             'E':E}
    y, merge_value= forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator)
    
    merge_value_arra.append(merge_value)