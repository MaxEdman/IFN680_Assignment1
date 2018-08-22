def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
            
        '''DONE - INSERT MISSING CODE HERE'''
        # Following 2 lines inserted by Max
        for i in range(0,len(w)):
            y = w[i] * (x**i)
    
        return y
    
    
    
def rmse(w):
    '''
    Compute and return the root mean squared error (RMSE) of the 
    polynomial defined by the weight vector w. 
    The RMSE is is evaluated on the training set (X,Y) where X and Y
    are the numpy arrays defined in the context of function 'task_1'.        
    '''
    Y_pred = fmodel(X, w)
    return np.sqrt(sum((Y_pred-X)**2).mean())