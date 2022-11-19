import numpy as np
import Atomic_Like_Potential as ALP


def S_Hartree_for_poisson_equation(r_grid, u_func, Numerov_backwards=True):
    #r_grid np.array-> [r_min, ..., r_max]
    #u_func np.array-> [u(r_min), ..., u(r_max)]
    #Numerov_backwards -> return s np.array in backwards order [s(r_max), ..., s(r_min)]
    #returns S array for 
    s_out= -1.0*np.divide(u_func**2.0, r_grid)
    if Numerov_backwards:
        s_out= np.flip(s_out)
    return list(s_out)

def K2_Hartree_for_poisson_equation(r_n,VH,n,**kwargs):#specific to schrodinger equation's
    #K2 as defined in the Numerov algorithm
    #for solving for uH function
    return 0.0e-8

def K2_SE_VH(r_n,VH_n,n,**kwargs):#specific to schrodinger equation's
    #K2 as defined in the Numerov algorithm
    #E energy eigenvalue of schrodinger's equation
    E=kwargs['E']
    Z=kwargs['Z']
    l=kwargs['l']
    return 2.0*(E - ALP.V_coulomb(r_n, Z) - ALP.V_angular(r_n,l) - VH_n)

def uH_func_boundery_condition(uH_func, r_grid, Q_max):
#fitting ax+b to uH_func to meet condtions uH_func(0) 0
#uH_func(r_max) = q_max -> Z
    b= -1.0*uH_func[0]
    a= (Q_max - uH_func[-1] - b)/r_grid[-1]
    return np.array(uH_func) +a*np.array(r_grid) +b

def uH_func_boundery_condition_forward(uH_func, r_grid, Q_max):
#fitting ax+b to uH_func to meet condtions uH_func(0) 0
#uH_func(r_max) = q_max -> Z
    a= (Q_max - uH_func[-1])/r_grid[-1]
    return np.array(uH_func) +a*np.array(r_grid)

def get_Vxc(r_grid, u_func):
    u= np.array(u_func)
    r= np.array(r_grid)
    nume= 3.0*u*u
    deno= 2.0*(np.pi**2.0)*r*r
    return -1.0*np.power(np.divide(nume,deno), (1.0/3.0))
