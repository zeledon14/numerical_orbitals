import numpy as np

def G(delta_x,K2, gamma=1.0):
    temp= (1.0 + (gamma*(delta_x**2.0)*K2)/12.0)
    return temp

def Numerov_backwards(y_func, r_grid,VH,K2,S,**kwargs):
    #integrates a function starting at R_N
    #finisdelta_xing at r_0 != 0.0
    #
    y_func_out= list(y_func)
    delta_x= kwargs['delta_x']
    for n_2, r_n_2 in enumerate(r_grid[2:]):
        K2n= K2(r_grid[n_2],VH[n_2],n_2,**kwargs)
        K2n_1= K2(r_grid[n_2+1],VH[n_2+1],n_2+1,**kwargs)
        K2n_2= K2(r_n_2,VH[n_2+1],n_2+1,**kwargs)

        S2n= S[n_2]#(r_grid[n_2],**kwargs)
        S2n_1= S[n_2+1]#(r_grid[n_2+1],**kwargs)
        S2n_2= S[n_2+2]#S(r_n_2,**kwargs)
        numerov_nume= (2.0*G(delta_x,K2n_1, gamma=-5.0)*y_func_out[n_2+1] - G(delta_x,K2n)*y_func_out[n_2] \
            + ((delta_x**2.0)/12.0)*(S2n + 10.0*S2n_1 + S2n_2))
        numerov_deno= G(delta_x,K2n_2)
        y_func_out.append((numerov_nume/numerov_deno))
    return np.flip(np.array(y_func_out)), np.flip(np.array(r_grid))

def Numerov_forward(y_func, r_grid_forw,VH,K2,S,**kwargs):
    #integrates a function starting at R_N
    #finisdelta_xing at r_0 != 0.0
    #
    y_func_out= list(y_func)
    delta_x= kwargs['delta_x']
    for n_2, r_n_2 in enumerate(r_grid_forw[2:]):
        K2n= K2(r_grid_forw[n_2],VH[n_2],n_2,**kwargs)
        K2n_1= K2(r_grid_forw[n_2+1],VH[n_2+1],n_2+1,**kwargs)
        K2n_2= K2(r_n_2,VH[n_2+1],n_2+1,**kwargs)

        S2n= S[n_2]#(r_grid_forw[n_2],**kwargs)
        S2n_1= S[n_2+1]#(r_grid[n_2+1],**kwargs)
        S2n_2= S[n_2+2]#S(r_grid_forw,**kwargs)
        numerov_nume= (2.0*G(delta_x,K2n_1, gamma=-5.0)*y_func_out[n_2+1] - G(delta_x,K2n_2)*y_func_out[n_2] \
            + ((delta_x**2.0)/12.0)*(S2n + 10.0*S2n_1 + S2n_2))
        numerov_deno= G(delta_x,K2n)
        y_func_out.append((numerov_nume/numerov_deno))
    return np.array(y_func_out), np.array(r_grid_forw)