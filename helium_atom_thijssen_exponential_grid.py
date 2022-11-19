import numpy as np
import matplotlib.pyplot as plt
import json
import os
plt.rcParams["figure.figsize"] = (25,10)

plt.rcParams.update({'font.size': 22})

colors='bgrykcm'
symbol='*-+^'

import utils
import utils_exponential_grid
from utils_exponential_grid import predictor_corrector_radial_shcrodinger_integrator_exponential_grid as integrator
from utils_exponential_grid import normlize_function_exponential_grid as normalizer
from utils_exponential_grid import predictor_corrector_radial_poisson_equation_electronic_potential_exponential_grid as poisson_integrator

occu_rule={1:{'n':1, 'l':0},
2:{'n':1,'l':0},
3:{'n':2,'l':0},
4:{'n':2,'l':0},
5:{'n':3,'l':1},
6:{'n':3,'l':1},
7:{'n':3,'l':1},
8:{'n':3,'l':1},
9:{'n':3,'l':1},
10:{'n':3,'l':1}}

max_energy_levels_by_l={0:2,1:3}

occupations_by_level=[2, 1]

#kwargs={'r_max':15.0,'grid_points':2000,'delta':0.0001,  
#        'l':1, 'Z':2.0, 'E':-0.5, 'max_numb_elec':5}
##        'r_N':15.0, 'delta_x':0.001}

kwargs={'r_max':5.0,'grid_points':3000,'delta':0.00001,  
        'l':1, 'Z':3.0, 'E':-0.5, 'max_numb_elec':sum(occupations_by_level), 
        'max_energy_level':len(occupations_by_level),
        'occu_rule':occu_rule, 'max_energy_levels_by_l':max_energy_levels_by_l,
        'occupations_by_level':occupations_by_level}
#        'r_N':15.0, 'delta_x':0.001}

#if the wave functions explode while r-> 0.0 make the energy mesh smaller

Energy_kwargs={'r_max':None, 'grid_points':100,'delta':0.001}
Energy_kwargs['r_max']= (0.5*kwargs['Z']**2 + 0.2)

exp_grid= utils_exponential_grid.get_exponential_grid_reverse(kwargs)
exp_grid_back= utils_exponential_grid.get_exponential_grid_reverse(kwargs)
#exp_grid=  utils.get_uniform_r_grid(**kwargs) #just testing
ener_grid= -1.0*np.array(utils_exponential_grid.get_exponential_grid_reverse(Energy_kwargs))
#ener_grid= [-0.7, -0.4]#just testing

#initial conditions
w10= 1.0e-10#u_hydr[0]
w20= (w10 - 1.2e-10)/(exp_grid[0] - exp_grid[1])#(u_hydr[1] - u_hydr[0])/(exp_grid[0] - exp_grid[1])

v_hart=None
v_xc=None
basis= utils.get_u_basis_set(kwargs, exp_grid, ener_grid, 
                                             w10,w20,integrator, normalizer, v_hart, v_xc)

E_befo=2.0
E= 1.0

while abs(E_befo-E) > 10e-6:
    E_befo=E

    u_dens= utils.get_u_dens(basis,kwargs)

    v_hart= utils_exponential_grid.get_V_hartree(exp_grid, u_dens, kwargs)
    
    v_xc= utils.get_v_xc(exp_grid, basis, kwargs)



    basis= utils.get_u_basis_set(kwargs, exp_grid, ener_grid, 
                                                 w10,w20,integrator, normalizer, v_hart, v_xc)
    E_hart= 0.5*utils_exponential_grid.integrate_functions_exponential_grid(exp_grid, v_hart, 
                                                                        np.array(basis[0]['u'])**2)
    E_xc= utils_exponential_grid.integrate_functions_exponential_grid(exp_grid, v_xc, 
                                                                        np.array(basis[0]['u'])**2)

    E= 2.0*basis[0]['E'] - E_hart - E_xc
    print('eigen ', basis[0]['E'])
    print('E_hart ', E_hart)
    print('E_xc ', E_xc)
    print('E ', E)
    print('________________________')