import numpy as np

def G(delta_x,K_sqrt, gamma=1.0):
    temp= (1.0 + (gamma*(delta_x**2.0)*K_sqrt)/12.0)
    return temp

def Radial_Schrodinger_Equation_Numerov_Backwards_Exp_Grid(u_func, r_exp_grid, VH, K_sqrt, kwargs):
    #r_exp_grid a list with the exponential grid points [r_0, r_1, ..., r_N_2, r_N_1]
    #u_func a list with the initial values of the u function
    #that is the eigen function of the radial schrodinger equation
    #u_func= [u(r_N_1), u(r_N_2)]
    print('here')
    delta_x= r_exp_grid[1:] - r_exp_grid[:-1]
    u_func_temp=len(r_exp_grid)*[1.0]
    u_func_temp[-1]= u_func[0]
    u_func_temp[-2]= u_func[1]
    delta=  kwargs['delta']
    for j in reversed(range(len(r_exp_grid)-2)):
        m=j
        m1=j+1
        m2=j+2
        K_sqrt_m= K_sqrt(r_exp_grid[m],VH[m],m,kwargs)
        K_sqrt_m1= K_sqrt(r_exp_grid[m1],VH[m1],m1,kwargs)
        K_sqrt_m2= K_sqrt(r_exp_grid[m2],VH[m2],m2,kwargs)

        numerov_nume= (2.0*G(delta_x[m1-1],K_sqrt_m1, gamma=-5.0)*u_func_temp[m1]*np.exp(-1.0*delta*m1*0.5) \
                       - G(delta_x[m2-1],K_sqrt_m2)*u_func_temp[m2]*np.exp(-1.0*delta*m2*0.5))

        numerov_deno= G(delta_x[m-1],K_sqrt_m)

        u_func_temp.append(np.exp(-1.0*delta*m*0.5)*numerov_nume/numerov_deno)

    u_func_temp= np.array(reversed(u_func_temp))
    

def Numerov_backwards(y_func, r_grid,VH,K_sqrt,S,**kwargs):
    #integrates a function starting at R_N
    #finisdelta_xing at r_0 != 0.0
    #
    y_func_out= list(y_func),
    delta_x= kwargs['delta_x']
    for n_2, r_n_2 in enumerate(r_grid[2:]):
        K_sqrtn= K_sqrt(r_grid[n_2],VH[n_2],n_2,**kwargs)
        K_sqrtn_1= K_sqrt(r_grid[n_2+1],VH[n_2+1],n_2+1,**kwargs)
        K_sqrtn_2= K_sqrt(r_n_2,VH[n_2+1],n_2+1,**kwargs)

        S2n= S[n_2]#(r_grid[n_2],**kwargs)
        S2n_1= S[n_2+1]#(r_grid[n_2+1],**kwargs)
        S2n_2= S[n_2+2]#S(r_n_2,**kwargs)
        numerov_nume= (2.0*G(delta_x,K_sqrtn_1, gamma=-5.0)*y_func_out[n_2+1] - G(delta_x,K_sqrtn)*y_func_out[n_2] \
            + ((delta_x**2.0)/12.0)*(S2n + 10.0*S2n_1 + S2n_2))
        numerov_deno= G(delta_x,K_sqrtn_2)
        y_func_out.append((numerov_nume/numerov_deno))
    return np.flip(np.array(y_func_out)), np.flip(np.array(r_grid))

def Numerov_forward(y_func, r_grid_forw,VH,K_sqrt,S,**kwargs):
    #integrates a function starting at R_N
    #finisdelta_xing at r_0 != 0.0
    #
    y_func_out= list(y_func)
    delta_x= kwargs['delta_x']
    for n_2, r_n_2 in enumerate(r_grid_forw[2:]):
        K_sqrtn= K_sqrt(r_grid_forw[n_2],VH[n_2],n_2,**kwargs)
        K_sqrtn_1= K_sqrt(r_grid_forw[n_2+1],VH[n_2+1],n_2+1,**kwargs)
        K_sqrtn_2= K_sqrt(r_n_2,VH[n_2+1],n_2+1,**kwargs)

        S2n= S[n_2]#(r_grid_forw[n_2],**kwargs)
        S2n_1= S[n_2+1]#(r_grid[n_2+1],**kwargs)
        S2n_2= S[n_2+2]#S(r_grid_forw,**kwargs)
        numerov_nume= (2.0*G(delta_x,K_sqrtn_1, gamma=-5.0)*y_func_out[n_2+1] - G(delta_x,K_sqrtn_2)*y_func_out[n_2] \
            + ((delta_x**2.0)/12.0)*(S2n + 10.0*S2n_1 + S2n_2))
        numerov_deno= G(delta_x,K_sqrtn)
        y_func_out.append((numerov_nume/numerov_deno))
    return np.array(y_func_out), np.array(r_grid_forw)