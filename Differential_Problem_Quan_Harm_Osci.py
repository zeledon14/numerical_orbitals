import numpy as np

def V(x):
    return (1.0/2.0)*x**2.0

def F1(x,E):
    return 2.0*(V(x) - E)


class DP_QHO():

    
    def __init__(self, param):
        """*grid:The grid.
            *k: 2D array with the k values with shape 
                (4, number of first order differential equations).
            *y: list of lists with answer to system of diff equations
                [m,i]
            *E: energy eigenvalue"""

        self.r=param['grid']
        self.y=param['y_arra']
        self.E=param['E']
        self.F1=F1(self.r, self.E)
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
                return h*F1(r, self.E)*self.y[0][i]
            elif j == 1 or j == 2:
                return h*F1((r+0.5*h), self.E)*(self.y[0][i] + 0.5*k[(j-1)][1])
            elif j == 3:
                return h*F1((r + h), self.E)*(self.y[0][i] + k[(j-1)][1])

    def FAM4(self, i, m, t):
        #return the value of the function for Adams Moulton degree 4
        if m==0:
            return self.y[1][(i-t)]
        elif m == 1:
            return F1(self.r[(i-t)], self.E)*self.y[0][(i-t)]

    def FpAM4(self, i, m):
        #return the value of the function for Adams Moulton degree 4
        #for the correction step only
        if m==0:
            return self.yp[1]
        elif m == 1:
            return F1(self.r[(i+1)], self.E)*self.yp[0]


