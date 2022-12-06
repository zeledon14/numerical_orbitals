import numpy as np

def f(x):
    return 2.0*np.exp(-1.0*x**2.0)*(2.0*x**2.0 - 1.0)


class DP_testing():

    
    def __init__(self, grid,y_arra):
        """*grid:The grid.
            *k: 2D array with the k values with shape 
                (4, number of first order differential equations).
            *y: list of lists with answer to system of diff equations
                [m,i]"""

        self.r=grid
        self.y=y_arra
        #initialize predictor y for every m equation
        self.yp=len(self.y)*[0.0]
        #grid spacing calculation
        if self.r[1]>self.r[0]:#forward case
            self.h= self.r[1:] - self.r[:-1]
        else:
            self.h= self.r[:-1] - self.r[1:]
        



    def k_for_rk4(self, i, k, j, m):
        """Calculates the k values for the RK4 integration
            *i: Current integration step.
            *k: 2D array with the k values with shape 
                (4, number of first order differential equations).
            *j: current k to be calcualted.
            *m:differential equation bo be solved"""
        h=self.h[i]
        r=self.r[i]
        if m == 0:
            if j == 0: #calculates the k1 value for rk4
                return h*self.y[1][i]
            elif j == 1 or j == 2:
                return h*(self.y[1][i] + 0.5*k[(j-1)][1])
            elif j == 3:
                return h*(self.y[1][i] + k[(j-1)][1])
        elif m == 1:
            if j == 0: #calculates the k1 value for rk4
                return h*f(r)
            elif j == 1 or j == 2:
                return h*f((r+0.5*h))
            elif j == 3:
                return h*f((r + h))

    def FAM4(self, i, m, t):
        #return the value of the function for Adams Moulton degree 4
        if m==0:
            return self.y[1][(i-t)]
        elif m == 1:
            return f(self.r[(i-t)])

    def FpAM4(self, i, m):
        #return the value of the function for Adams Moulton degree 4
        #for the correction step only
        if m==0:
            return self.yp[1]
        elif m == 1:
            return f(self.r[(i+1)])


