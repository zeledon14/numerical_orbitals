import numpy as np
#atomic like potentials
def V_coulomb(r_n, Z):
    #Coulomb potential produced by Z protons
    # acting on a electron on atomic units
    #Z atomic number
    #r_n evaluation point or array of evaluation points
    return (-1.0*Z)/r_n

def V_angular(r_n,l):
    #potential produced by angular momentum 
    #r_n evaluation point or array of evaluation points
    #l angular number 
    return (l*(l+1.0))/(2.0*(r_n**2.0))

def K_sqrt_exp_grid_radi_schr(r_n, VH_n, j,kwargs):
    #specific to radial schrodinger equation's in 
    #K2 as defined in the Numerov algorithm
    #E energy eigenvalue of schrodinger's equation
    E=kwargs['E']
    Z=kwargs['Z']
    l=kwargs['l']
    rp=kwargs['rp']
    delta=kwargs['delta']
    out= -1.0*(delta**2.0)/4.0 - 2.0*((rp*delta)**0.0)*np.exp(2.0*delta*j)*(V_coulomb(r_n, Z) + V_angular(r_n,l) - E)
    return out

def K2_atomic_radial(r_n,VH_n,n,**kwargs):#specific to schrodinger equation's
    #K2 as defined in the Numerov algorithm
    #E energy eigenvalue of schrodinger's equation
    E=kwargs['E']
    Z=kwargs['Z']
    l=kwargs['l']
    return 2.0*(E - V_coulomb(r_n, Z) - V_angular(r_n,l))

def V_eff(r_n, Z, l, E):
    #effective potential for intergration of SE using predictor corrector
    return 2.0*(V_coulomb(r_n, Z) + V_angular(r_n,l) - E)

def V_eff_testing(r_n, Z, l, E):
    #potential for the differential equation d^2y/dx^2 = (4(x-5)^2-2)y
    #here y= exp(-(x-5)^2)
    return (4.0*(r_n-5.0)**2.0 - 2.0)


def V_eff_exponential_grid(delta, rp, j, r_n, Z, l, E):
    #effective potential for intergration of SE using predictor corrector in exponential grid
    return (delta**2.0)/4.0 + np.exp(2.0*delta*j)*(rp**2.0)*(delta**2.0)*2.0*(V_coulomb(r_n, Z) + V_angular(r_n,l) - E)

#def S_atomic_radial(r_n,**kwargs):#specific to schrodinger equation's
#    #S as defined in the Numerov algorithm
#    return 0.0