import os
import json
import numpy as np
from pytools import T

def get_uniform_r_grid(**kwargs):
    r_grid=[kwargs['r_N']]
    r_n= kwargs['r_N']- kwargs['delta_x']
    while r_n > 0.0:
        r_grid.append(r_n)
        r_n-= kwargs['delta_x']
    return r_grid

def get_uniform_grid(delta, max_valu):
    grid=[max_valu]
    r_n= max_valu- delta
    while r_n > 0.0:
        grid.append(r_n)
        r_n-= delta
    return grid

"""def get_rp(r_max, grid_points, delta):
    return r_max/(np.exp(delta*grid_points) - 1.0)

def get_exponential_grid(delta=0.001, rp=0.9, grid_points=20):
    return list(rp*(np.exp(delta*np.array([*range(grid_points)])) - 1.0))
    #exp_grid= [rp*(np.exp(i*delta)-1.0) for i in range(grid_points)]
    #return exp_grid"""

def get_nodes_information(f, x):
    #input: f -> an f(x) array with the functions values
    #input: x -> array with the x values
    #output: number of nodes in f (wave function)
    #and the (x1, x2)_i values in between the i node is
    #x1 < x2
    number_of_nodes=0
    nodes_positions=[]
    nodes_indices=[]
    for i, _ in enumerate(f[:-1]):
        if int(np.sign(f[i])) != int(np.sign(f[i+1])):
            number_of_nodes+=1
            if x[i] < x[i+1]:
                nodes_positions.append((x[i], x[i+1]))
                nodes_indices.append((i,i+1))
            else:
                nodes_positions.append((x[i+1], x[i]))
                nodes_indices.append((i+1, i))
    return number_of_nodes, nodes_positions, nodes_indices

def false_position(Ea,Eb,fa,fb):
    #for root finding algorithm
    return (Eb*fa - Ea*fb)/(fa - fb)

def U_Hydrogen(r_grid):
    if isinstance(r_grid, list):
        temp=np.array(r_grid)
        return np.array(temp)*np.exp(-1.0*temp)
    else:
        return np.array(r_grid)*np.exp(-1.0*r_grid) 

def integrate_f(f,delta_x):
    f_forw= np.array(f)[1:]
    f_back= np.array(f)[:-1]
    return np.sum((delta_x/2.0)*(f_forw+f_back))

def normalize_u_function(u_func, delta_x):
    temp= np.squeeze(np.array(u_func))**2.0
    I=np.sum((delta_x/2.0)*(temp[1:]+temp[:-1]))
    I=I**0.5
    #self.radi_eige_func=(1.0/I)*(np.squeeze(np.array(self.u_func))/np.squeeze(np.array(self.r_grid)))
    #self.u_func_normalized=True
    return (1.0/I)*np.squeeze(np.array(u_func))

def copy_kwargs(kwargs):
    kwargs_copy= {key:valu for key,valu in kwargs.items()}
    return kwargs_copy

def find_eigenvalue_secant_method(i_nodes_positions, kwargs, grid,
                                    integrator, normalizer,
                                    w10, w20, 
                                    v_h, v_x,
                                    N_max=100,tolerance=1.0e-10):
    #secant method
    temp_nodes_posi= i_nodes_positions
    i=0


    p0= temp_nodes_posi[0]
    kwargs_temp=copy_kwargs(kwargs)
    kwargs_temp['E']=p0
    u_func= integrator(grid, w10,w20, kwargs_temp, v_h=v_h, v_x= v_x)
    #u_func= normalizer(grid, u_func)
    q0= u_func[-1]

    p1= temp_nodes_posi[1]
    kwargs_temp=copy_kwargs(kwargs)
    kwargs_temp['E']=p1
    u_func= integrator(grid, w10,w20, kwargs_temp, v_h=v_h, v_x= v_x)
    #u_func= normalizer(grid, u_func)
    q1= u_func[-1]
    #print('i ', i, 'p0 ', p0, 'p1 ', p1, 'q0 ', q0, 'q1 ', q1)
    i+=1

    while i < N_max:
        #print(i)
        p= p1 - q1*(p1-p0)/(q1-q0)
        if abs(p-p1) < tolerance:
            break
        p0=p1
        q0=q1
        p1=p
        kwargs_temp=copy_kwargs(kwargs)
        kwargs_temp['E']=p1
        u_func= integrator(grid, w10,w20, kwargs_temp, v_h=v_h, v_x= v_x)
        #u_func= normalizer(grid, u_func)
        q1= u_func[-1]
        #print(q1)
        #print('i ', i, 'p0 ', p0, 'p1 ', p1, 'q0 ', q0, 'q1 ', q1)
        i+=1
    kwargs_temp=copy_kwargs(kwargs)
    kwargs_temp['E']=p
    u_func= integrator(grid, w10,w20, kwargs_temp, v_h=v_h, v_x= v_x)
    u_func= normalizer(grid, u_func)
    #return {'u_func_norm':u_func, 'E':p}
    return u_func, p, kwargs_temp, grid

def get_u_basis_set(kwargs, grid, ener_grid, w10,w20,
                    integrator, normalizer, v_h, v_x):

    occu_rule= kwargs['occu_rule']
    l_arra=[*range(occu_rule[kwargs['max_numb_elec']]['l'] + 1)]
    max_energy_levels_by_l= kwargs['max_energy_levels_by_l']
    nodes_positions_by_l={}
    for l in l_arra:
        u0_E=[]# u_func(r=0) as function of energy
        for E in ener_grid:
            kwargs_temp=copy_kwargs(kwargs)
            kwargs_temp['E']=E
            kwargs_temp['l']=l
            u_func= integrator(grid, w10,w20, kwargs_temp, v_h=v_h, v_x= v_x)
            u_func_norm= normalizer(grid, u_func)
            u0_E.append(u_func_norm[-1])
        number_of_nodes, nodes_positions= get_nodes_information(u0_E, ener_grid)
        nodes_positions_by_l[l]=nodes_positions
    max_ener_levels= occu_rule[kwargs['max_numb_elec']]['n']

    basis=[]
    i_energy_level=0
    for l in l_arra:
        for i_nodes_positions in nodes_positions_by_l[l][:max_energy_levels_by_l[l]]:
            if i_energy_level < kwargs['max_energy_level']:
                kwargs['l']=l
                #print(i_nodes_positions)
                u_func, p, p_kwargs, p_grid= find_eigenvalue_secant_method(i_nodes_positions, kwargs, grid, integrator, normalizer,
                                                                w10, w20, v_h, v_x)
                basis.append({'u':u_func,
                                'E':p,
                                'l':l,
                                'grid':p_grid,
                                'kwargs':kwargs})
                i_energy_level+=1
            else:
                break
    return basis

def get_u_dens(basis,kwargs):
    u_dens=np.zeros_like(basis[0]['u'])
    for i, occu_numb in enumerate(kwargs['occupations_by_level']):
        u_dens+= occu_numb*np.power(np.array(basis[i]['u']), 2.0)
    return u_dens

def get_v_xc(grid, basis, kwargs):
    v_xc=np.zeros_like(basis[0]['u'])
    c=(3.0/(2.0*np.pi**2.0))
    grid_sqr= np.power(np.array(grid), 2.0)
    for i, occu_numb in enumerate(kwargs['occupations_by_level']):
        v_xc+= 1.0*np.power(c*np.divide(np.power(np.array(basis[i]['u']), 2.0),grid_sqr), (1.0/3.0))
    v_xc[-1]= v_xc[-2]
    return v_xc
