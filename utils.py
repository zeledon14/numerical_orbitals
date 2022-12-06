import numpy as np

def get_nodes_information(f, x):
    """
    *f: an f(x) array with the functions values.
    *x: array with the x values.

    *returns:
        *number of nodes in f (wave function)
        *(x1, x2)_i values in between the i node is
                    x1 < x2
        *(i, i+1) indices between which the node is."""
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

def forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator):
    """
    *param_f: dictionary with the parameters for the forward integration.
    *param_b: dictionary with the parameters for the backwards integration.
    *diff_prob: differential problem constructor.
    *integratior: integrator to integrate the differential problem.
    
    *returns: 
        *y: the integrated function over the grid provided in param_f as order in param_f
        *merge_value: """
    DP_f=diff_prob(param_f)
    y_f=integrator(DP_f)
    y_f=y_f[0]
    try:
        grid= param_f['grid']
    except OSError:
        print('No grid inside the param_f dictionary')
    #finding the turning points where the F1= V-E changes sign 
    #at that point is where we merge the forward solution with the backwards solution
    number_of_nodes, nodes_positions, nodes_indices= get_nodes_information(DP_f.F1, grid)

    DP_b=diff_prob(param_b)
    y_b=integrator(DP_b)
    y_b=np.flip(y_b[0])

    y= np.array(y_f)
    #odd case of eigenfunation
    if np.sign(y_f[(nodes_indices[0][0])]) != np.sign(y_b[(nodes_indices[0][0])]):
        y_b= np.multiply(-1.0, y_b)
    #merge_value=(y_f[(nodes_indices[0][0])] - y_b[(nodes_indices[0][0])])
    y_fp= ((y_f[(nodes_indices[0][0])] - y_f[(nodes_indices[0][0] - 1)])
            /(grid[(nodes_indices[0][0])] - grid[(nodes_indices[0][0] - 1)]))
    y_bp= ((y_b[(nodes_indices[0][0])] - y_b[(nodes_indices[0][0] - 1)])
            /(grid[(nodes_indices[0][0])] - grid[(nodes_indices[0][0] - 1)]))
    merge_value=(y_fp - y_bp)
    y[nodes_indices[0][0]:]= y_b[nodes_indices[0][0]:]       
    merge_value=(y_f[(nodes_indices[0][0])] - y_b[(nodes_indices[0][0])])
    return y, merge_value

def normlize_function(grid, function):
    """Normalize the input function over the grid
       *grid: array with the grid over which the function is defined
       *function: array with the function to normalize
       
       *output
            *func_norm: array with the normalized function"""
    if grid[1]>grid[0]:#forward case
        h= grid[1:] - grid[:-1]
    else:
        h= grid[:-1] - grid[1:]

    func_sqrt= np.power(function, 2.0)
    I= np.sum(np.multiply((h/2.0), np.add(func_sqrt[:-1],func_sqrt[1:])))
    I=I**0.5
    func_norm= np.divide(function,I)
    return func_norm

def find_eigenvalue_secant_method(i_nodes_positions,grid, 
                                    y1_f,y2_f,y1_b,y2_b,
                                    diff_prob, integrator,
                                    N_max=100,tolerance=1.0e-10):
    #secant method
    temp_nodes_posi= i_nodes_positions
    i=0

    p0= temp_nodes_posi[0]
    param_f={'grid':grid,
             'y_arra':[[y1_f],[y2_f]],
             'E':p0}
    param_b={'grid':np.flip(grid),
             'y_arra':[[y1_b],[y2_b]],
             'E':p0}

    y_p0, q0= forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator)

    p1= temp_nodes_posi[1]
    param_f={'grid':grid,
             'y_arra':[[y1_f],[y2_f]],
             'E':p1}
    param_b={'grid':np.flip(grid),
             'y_arra':[[y1_b],[y2_b]],
             'E':p1}

    y_p1, q1= forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator)
    #u_func= normalizer(grid, u_func)
    i+=1
    while i < N_max:
        #print(i)
        p= p1 - q1*(p1-p0)/(q1-q0)
        #print(abs(p-p1))
        if abs(p-p1) < tolerance:
            break
        p0=p1
        q0=q1
        p1=p
        param_f={'grid':grid,
                'y_arra':[[y1_f],[y2_f]],
                'E':p1}
        param_b={'grid':np.flip(grid),
                'y_arra':[[y1_b],[y2_b]],
                'E':p1}
        y_p1, q1= forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator)

        i+=1
    param_f={'grid':grid,
            'y_arra':[[y1_f],[y2_f]],
            'E':p}
    param_b={'grid':np.flip(grid),
            'y_arra':[[y1_b],[y2_b]],
            'E':p}
    y, _= forward_backwards_integration_and_merge_value(param_f, param_b, diff_prob, integrator)
    y= normlize_function(grid, y)
    #return {'u_func_norm':u_func, 'E':p}
    return y, p

def f_derivative(grid, function):
    """Normalize the input function over the grid
    *grid: array with the grid over which the function is defined
    *function: array with the function to normalize
    
    *output
            *derivative: array with the derivative of the input function"""
    
    if grid[1]>grid[0]:#forward case
        h= grid[1:] - grid[:-1]
        diff= function[1:] - function[:-1]
    else:
        h= grid[:-1] - grid[1:]
        diff= function[:-1] - function[1:]
    derivative= np.divide(diff,h)
    return derivative

def is_function_smooth(function):
    """Returns True if the input function is smooth under the criteria
    *function: array with the function to check
    
    *output
            *True if function is smooth False if is not."""
    output=True
    temp= np.abs(np.subtract(function[1:],function[:-1]))
    max_valu= np.max(temp)
    aver= float((np.sum(temp)-max_valu))/float((len(temp) -1)) #average without max value
    if max_valu> 150*aver:
        output=False
    return output
