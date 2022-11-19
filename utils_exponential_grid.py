import numpy as np
from Atomic_Like_Potential import V_eff
#from Atomic_Like_Potential import V_eff_testing as V_eff


def predictor_corrector_radial_shcrodinger_integrator_exponential_grid(grid, w10,w20, kwargs,v_h=None, v_x= None):
    E= kwargs['E']
    #h= kwargs['delta_x']
    #h_arra= np.array(grid[1:]) - np.array(grid[:-1])
    h_arra= np.array(grid[:-1]) - np.array(grid[1:])
    Z= kwargs['Z']
    l= float(kwargs['l'])
    if v_h is None:
        v_h= np.zeros_like(grid)
    
    if v_x is None:
        v_x= np.zeros_like(grid)

    w1=[w10]
    w2=[w20]

    for i,ri in enumerate(grid[:3]):
        h= h_arra[i]
        k1=4*[0.0]
        k2=4*[0.0]
        vj0= V_eff(ri,Z,l,E) + v_h[i] + v_x[i]
        vj12= V_eff(ri + 0.5*h,Z,l,E) + 0.5*(v_h[i] + v_x[i] + v_h[i+1] + v_x[i+1])
        vj3= V_eff(ri + h,Z,l,E) + v_h[i+1] + v_x[i+1]
        for j in range(4):
            
            if j == 0:
                k1[j]=h*(w2[i])
                k2[j]= h*vj0*w1[i]
            elif j == 1 or j == 2:
                k1[j]=h*(w2[i] + 0.5*k2[j-1])
                k2[j]= h*vj12*(w1[i] + 0.5*k1[j-1])
            elif j == 3:
                k1[j]=h*(w2[i] + k2[j-1])
                k2[j]= h*vj3*(w1[i] + k1[j-1])
        w1.append( w1[i] + (1.0/6.0)*(k1[0] + 2.0*k1[1] + 2.0*k1[2] + k1[3]))
        w2.append( w2[i] + (1.0/6.0)*(k2[0] + 2.0*k2[1] + 2.0*k2[2] + k2[3]))

    for i in range(3,(len(grid)-1)):
        h= h_arra[i]
        ri_plus_1= grid[i+1]
        ri= grid[i]
        r1= grid[i-1]
        r2= grid[i-2]
        r3= grid[i-3]
        
        Vint_i_plus_1= v_h[i+1] + v_x[i+1]
        Vint_i= v_h[i] + v_x[i]
        Vint_1= v_h[i-1] + v_x[i-1]
        Vint_2= v_h[i-2] + v_x[i-2]
        Vint_3= v_h[i-3] + v_x[i-3]

        wp1= w1[i] + h*(55.0*w2[i] - 59.0*w2[i-1]
                       +37.0*w2[i-2] - 9.0*w2[i-3])/24.0

        wp2= w2[i] + h*(55.0*(V_eff(ri,Z,l,E) + Vint_i)*w1[i] - 59.0*(V_eff(r1,Z,l,E) + Vint_1)*w1[i-1]
                       +37.0*(V_eff(r2,Z,l,E)+Vint_2)*w1[i-2] - 9.0*(V_eff(r3,Z,l,E)+Vint_3)*w1[i-3])/24.0

        wc1= w1[i] + h*(9.0*wp2 + 19.0*w2[i]
                       -5.0*w2[i-1] + w2[i-2])/24.0

        wc2= w2[i] + h*(9.0*(V_eff(ri_plus_1,Z,l,E)+Vint_i_plus_1)*wc1 + 19.0*(V_eff(ri,Z,l,E)+Vint_i)*w1[i]
                       -5.0*(V_eff(r1,Z,l,E)+Vint_1)*w1[i-1] + (V_eff(r2,Z,l,E)+Vint_2)*w1[i-2])/24.0

        w1.append(wc1)
        w2.append(wc2)
    return w1

def normlize_function_exponential_grid(expo_grid, function):
    h_arra= np.array(expo_grid[:-1]) - np.array(expo_grid[1:])

    func_sqrt= np.array(function)**2.
    I= np.sum((h_arra/2.0)*(func_sqrt[:-1]+func_sqrt[1:]))
    I=I**0.5
    func_norm= function/I
    return func_norm

def get_rp(kwargs):
    r_max=kwargs['r_max']
    grid_points=kwargs['grid_points']  
    delta=kwargs['delta'] 
    return r_max/(np.exp(delta*grid_points) - 1.0)

def get_exponential_grid_reverse(kwargs):#delta=0.001, rp=0.9, grid_points=20):
    rp=get_rp(kwargs)
    grid_points=kwargs['grid_points']  
    delta=kwargs['delta'] 
    exp_grid= list(rp*(np.exp(delta*np.array([*range(grid_points)])) - 1.0))
    h_temp= exp_grid[-1] - exp_grid[-2]
    for i in range(0, 4):
        exp_grid.append(exp_grid[-1] + h_temp)
    exp_grid= list(np.array(exp_grid) + 1.10e-12)
    exp_grid= list(np.array(exp_grid))
    exp_grid.reverse()
    #exp_grid= exp_grid[:-1]
    #exp_grid= [rp*(np.exp(i*delta)-1.0) for i in range(grid_points)]
    return exp_grid


def predictor_corrector_radial_poisson_equation_electronic_potential_exponential_grid(grid, u_dens, 
    w10=0.0,w20=1.0, backward_grid= True):
    #backward_grid the default is true becuase for the Schrodinger equation grid[0] = r_max while grid[-1]= r_min
    #also u_dens[0] is the value of u_dens at r_max, and u_dens[-1] is the value of u_dens at r_min
    #FOR THE POISSON EQUATION INTEGRATION, THE ALGORTIHM FLIPS THE BACKWARD GRID BECAUSE THE BOUNDARY CONDITION IS EASIER TO SET
    #FOR THE CASE GRID[0]=R_MIN
    h_arra= np.array(grid[:-1]) - np.array(grid[1:])
    
    if backward_grid:
        grid= np.flip(np.array(grid))
        u_dens= np.flip(np.array(u_dens))
        h_arra= np.flip(np.array(h_arra))
    #f2= -1.0*np.divide(np.power(u_dens,2.0),grid)
    f2= -1.0*np.divide(u_dens,grid)
    w1=[w10]
    w2=[w20]

    #for i,ri in enumerate(grid[:-1]):#use with only rk4
    for i,ri in enumerate(grid[:3]):
        h= h_arra[i]
        k1=4*[0.0]
        k2=4*[0.0]
        vj0= f2[i]#V_eff(ri,Z,l,E)
        m= (f2[i+1]-f2[i])/h
        b= ((f2[i+1]+f2[i])-m*(2.0*ri + h))/2.0
        vj12= m*(ri +  0.5*h) + b#V_eff(ri + 0.5*h,Z,l,E)
        vj3= f2[i+1]#V_eff(ri + h,Z,l,E)
        for j in range(4):
            
            if j == 0:
                k1[j]=h*(w2[i])
                k2[j]= h*vj0
            elif j == 1 or j == 2:
                k1[j]=h*(w2[i] + 0.5*k2[j-1])
                k2[j]= h*vj12
            elif j == 3:
                k1[j]=h*(w2[i] + k2[j-1])
                k2[j]= h*vj3
        w1.append( w1[i] + (1.0/6.0)*(k1[0] + 2.0*k1[1] + 2.0*k1[2] + k1[3]))
        w2.append( w2[i] + (1.0/6.0)*(k2[0] + 2.0*k2[1] + 2.0*k2[2] + k2[3]))

    for i in range(3,(len(grid)-1)):
        h= h_arra[i]
        f2_i_plus_1=f2[i+1]
        f2i=f2[i]
        f21=f2[i-1]
        f22=f2[i-2]
        f23=f2[i-3]
        #r4= grid[i-4]
        wp1= w1[i] + h*(55.0*w2[i] - 59.0*w2[i-1]
                       +37.0*w2[i-2] - 9.0*w2[i-3])/24.0

        wp2= w2[i] + h*(55.0*f2i - 59.0*f21
                       +37.0*f22 - 9.0*f23)/24.0

        wc1= w1[i] + h*(9.0*wp2 + 19.0*w2[i]
                       -5.0*w2[i-1] + w2[i-2])/24.0

        wc2= w2[i] + h*(9.0*f2_i_plus_1 + 19.0*f2i
                       -5.0*f21 + f22)/24.0

        w1.append(wc1)
        w2.append(wc2)
    w1.reverse()
    return w1

def set_ue_boundery_condition(ue, grid, Q_max, backward_grid= True):
    if backward_grid:
        a= (Q_max - ue[0])/grid[0]
    return np.array(ue) + np.multiply(a,np.array(grid))

def get_V_hartree(grid, u_dens, kwargs):

    ue= predictor_corrector_radial_poisson_equation_electronic_potential_exponential_grid(grid,u_dens)
    ue=set_ue_boundery_condition(ue, grid, kwargs['max_numb_elec'])
    v_hart= np.divide(ue,grid)
    v_hart[-1]= v_hart[-2]#correction on v_hart[-1] introduced by the boundary condition
    return v_hart

def integrate_functions_exponential_grid(expo_grid, g,f):
    h_arra= np.array(expo_grid[:-1]) - np.array(expo_grid[1:])

    temp= np.multiply(np.array(f),np.array(g))
    I= np.sum((h_arra/2.0)*(temp[:-1]+temp[1:]))

    
    return I
