import numpy as np

def predictor_corrector_RK4_Adams_Moulton4(DP) -> list:
    #number of first order ODE
    numb= len(DP.y)
    #h_arra is the array with the spaces in between grid points
    #declaration of the k's for RK$ 
    k= np.zeros((4,numb), dtype=float)#4 because it is RK of degree 4
    #loop over the grid
    #first loop over first 4 places on the grid using RK4
    for i, ri in enumerate(DP.r[:4]):
        #loop over the 4 k's in RK4
        for j in range(4):
            #loop over number of first order ODE
            for m in range(numb):
                k[j,m]= DP.k_for_rk4(i, k, j, m)
                #k[j,m]= functions[m](ri, i, h_arra[i], k, j, y_arra)
        #make next step in the solution 
        for m in range(numb):
            DP.y[m].append(DP.y[m][i] + (1.0/6.0)*(k[0,m] + 2.0*k[1,m] + 2.0*k[2,m] + k[3,m]))
    #second loop from 4 place in grid to the end
    for ip, ri in enumerate(DP.r[4:-1]):
        i=ip+4#integration step
        for m in range(numb):
            DP.yp[m]= DP.y[m][i]  + (DP.h[i]/24.0)*(55.0*DP.FAM4(i,m,0) - 59.0*DP.FAM4(i,m,1)
                                     + 37.0*DP.FAM4(i,m,2) - 9.0*DP.FAM4(i,m,3))

        for m in range(numb):
            DP.y[m].append(DP.y[m][i]  + (DP.h[i]/24.0)*(9.0*DP.FpAM4(i,m) + 19.0*DP.FAM4(i,m,0)
                                    - 5.0*DP.FAM4(i,m,1) + DP.FAM4(i,m,2)))
    return np.array(DP.y)

def RK4(DP) -> list:
    """
    *functions: a list of Differential_Function objects."""
    #number of first order ODE
    numb= len(DP.y)
    #h_arra is the array with the spaces in between grid points
    #declaration of the k's for RK$ 
    k= np.zeros((4,numb), dtype=float)#4 because it is RK of degree 4
    #loop over the grid
    for i, ri in enumerate(DP.r[:-1]):
        #loop over the 4 k's in RK4
        for j in range(4):
            #loop over number of first order ODE
            for m in range(numb):
                k[j,m]= DP.k_for_rk4(i, k, j, m)
                #k[j,m]= functions[m](ri, i, h_arra[i], k, j, y_arra)
        #make next step in the solution 
        for m in range(numb):
            DP.y[m].append(DP.y[m][i] + (1.0/6.0)*(k[0,m] + 2.0*k[1,m] + 2.0*k[2,m] + k[3,m]))
    return np.array(DP.y)