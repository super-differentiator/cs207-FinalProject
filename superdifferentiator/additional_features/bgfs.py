from superdifferentiator.forward.functions import X
import numpy as np
import math



def bgfs(f, init_x, accuracy = 1e-8, alphas =[.00001,.00005,.0001,.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5],
         max_iter = 1000,verbose = False):
    
    x_current = [X(init_x[i],'x'+str(i)) for i in  range(len(init_x))]
    B = np.eye(len(init_x))
    
    for i_iter in range(max_iter):
        func = f(x_current)
        f_jac = func.jacobian()[1][0].T
        p = -np.linalg.pinv(B) @f_jac
        
        f_minmin = 10**13
        alpha_minmin = .00001
        for alpha in alphas:
            f_min = f([x_current[i] + alpha * p[i,0] for i in range(len(x_current))])
            if f_min.val[0] < f_minmin:
                f_minmin = f_min.val[0]
                alpha_minmin = alpha
        
        
        if math.isnan(float(p[0])):
            print("It doesn't converge using bgfs! Please use some other methods or check if a maximum/minimum exists!")
            return [x_current[i].val[0] for i in range(len(x_current))]
            
        s = alpha_minmin * p
        if np.linalg.norm(s) <= accuracy:
            print('Since the shifted distance is smaller than {}, we stop the loop at {}th iteration.'.format(accuracy,i_iter))
            break
        x_current = [x_current[i] + alpha_minmin * p[i,0] for i in range(len(x_current))]
        if verbose:
            print('At {}th iteration, current x value is: '.format(i_iter),[x_current[i].val[0] for i in range(len(x_current))],
                  ', the step taken is: ',s,'.\n' )
        y_current = f(x_current).jacobian()[1][0].T - f_jac
        
        
        B += y_current.dot(y_current.T)/(y_current.T.dot(s)) - B.dot(s).dot(s.T).dot(B.T)/(s.T.dot(B).dot(s))
        
    final_del = [x_current[i].val[0] for i in range(len(x_current))]
    if i_iter == max_iter-1:
        print("Notice that the change in distance of x is still larger than input accuracy! It probably doesn't converge using bgfs! ")
    if verbose:
        print('The final value we get is',final_del,'.')
    return final_del
    

