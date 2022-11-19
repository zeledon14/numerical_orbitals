import os
import json
import numpy as np

class Radial_Eigenfunction():
    def __init__(self, Ea=None, Eb=None, **kwargs):
        self.Ea= Ea
        self.Eb= Eb
        self.r_N= kwargs['r_N']
        self.delta_x=kwargs['delta_x']
        self.l=kwargs['l']
        self.Z=kwargs['Z']
        self.E=None
        self.u_func_normalized=False
        self.u_func=None
        self.radi_eige_func=None #normalized wave function
            
    def set_u_function(self,u_func):
        self.u_func= u_func
        
    def set_r_grid(self, r_grid):
        self.r_grid= r_grid
        
    def check_if_eigenvalue(self):
        if abs(self.Ea - self.Eb) > 1e-9:
            ##print('here1')
            return False
        else:
            #print('here2')
            self.E= (self.Ea + self.Eb)/2.0
            return True
        
    def bild_normalized_radi_eige_func(self):
        temp= np.squeeze(np.array(self.u_func))**2.0
        I=np.sum((self.delta_x/2.0)*(temp[1:]+temp[:-1]))
        I=I**0.5
        self.radi_eige_func=(1.0/I)*(np.squeeze(np.array(self.u_func))/np.squeeze(np.array(self.r_grid)))
        self.u_func_normalized=True
        self.u_func= (1.0/I)*np.squeeze(np.array(self.u_func))
    
    def save_as_json(self, path):
        temp={'Ea':self.Ea, 
                'Eb':self.Eb,
                'r_N':self.r_N,
                'h':self.delta_x,
                'l':self.l,
                'Z':self.Z,
                'E':self.E,
                'U_normalized':self.u_func_normalized,
                'r_grid':list(self.r_grid),
                'U':list(self.u_func),
                'radi_eige_func':list(self.radi_eige_func)}
        json.dump(temp, open(path, 'w'))
    
    def restore_from_json(self, path):
        temp= json.load(open(path, 'r'))
        self.Ea= temp['Ea']
        self.Eb= temp['Eb']
        self.r_N= temp['r_N']
        self.delta_x=temp['delta_x']
        self.l=temp['l']
        self.Z=temp['Z']
        self.E=temp['E']
        self.u_func_normalized=temp['u_func_normalized']
        self.r_grid=np.array(temp['r_grid'])
        self.u_func=np.array(temp['u_func'])
        self.radi_eige_func=np.array(temp['radi_eige_func'])