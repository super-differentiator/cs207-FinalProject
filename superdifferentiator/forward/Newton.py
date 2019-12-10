import numpy as np
from superdifferentiator.forward.functions import X
from superdifferentiator.forward.Vector import Vector



def Newton(f:list, init_x= None, accuracy = 1e-8, max_iter = 10000, verbose = True):
    if init_x is None:
        init_x = [1.0]*len(f)
    

    x_current = [X(init_x[i],'x'+str(i)) for i in  range(len(init_x))]
    
    
    for i_iter in range(max_iter):
        
        k = Vector([func(x_current) for func in f]) #k is the evaluated vector function
        
        reduction = np.linalg.pinv(k.jacobian()[1]).dot(k.val)
        x_current = [x_current[i] - float(reduction[0,i]) for i in range(len(x_current))]
    
        f_vec = Vector([func(x_current) for func in f]).val
        f_val = np.linalg.norm(f_vec)
        if verbose:
            print('At {}th iteration, current x value is: '.format(i_iter),[x_current[i].val[0] for i in range(len(x_current))],
                  ', the value of function is: ',f_val,'.\n' )
        if f_val < accuracy:
            print('Since the shifted distance is smaller than {}, we stop the loop at {}th iteration.'.format(accuracy,i_iter))
            break
    final_del = [x_current[i].val[0] for i in range(len(x_current))]
    if verbose:
        print('The final value we get is',final_del,'.')
    if i_iter == max_iter-1:
        print("Notice that the change in distance of x is still larger than input accuracy! The function value is currently {}. The result probably didn't converge!".format(f_vec))
    return final_del












   
