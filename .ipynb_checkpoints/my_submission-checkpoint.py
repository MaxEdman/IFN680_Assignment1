'''

2018 Assigment One : Differential Evolution
    
Scafolding code

Complete the missing code at the locations marked 
with 'INSERT MISSING CODE HERE'

To run task_2 you will need to download an unzip the file dataset.zip

If you have questions, drop by in one of the pracs on Wednesday 
     11am-1pm in S503 or 3pm-5pm in S517
You can also send questions via email to f.maire@qut.edu.au


'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    #................................................................
    
    n_dimensions = len(bounds) # dimension of the input space of 'fobj'
    #    This generates our initial population of 10 random vectors. 
    #    Each component x[i] is normalized between [0, 1]. 
    
    
    # Initialise an initial population of normalized random values. 
    # 2D-array with dimensions popsize * n_dimensions. For task_1 this is 6 * 20.
    population = np.random.rand(popsize, n_dimensions)
    #print(population)
    
    # Denormalizes the parameters to the corresponding values.
    # Need further commenting explanations of these steps.
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    denormalized_population = min_b + (population * diff)
    cost = np.asarray([fobj(i) for i in denormalized_population])
    
    best_idx = np.argmin(cost)
    best = denormalized_population[best_idx]
    
    if verbose:
        print(
        '** Lowest cost in initial population = {} '
        .format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))    
        
        for j in range(popsize) :
            #................................................................
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace = False)]
            # Creates a mutant vector and clips the entries to the interval [0,1]
            mut = np.clip(a + mut * (b - c), 0, 1)
            
            cross_points = np.random.rand(n_dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dimensions)] = True
            trial = np.where(cross_points, mut, population[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            #................................................................

            #................................................................
            if f < cost[j]:
                cost[j] = f
                population[j] = trial
                if f < cost[best_idx]: #and all(bounds[k][0]<=trial_denorm[k]<=bounds[k][1] for k in range(len(trial_denorm))):
                    best_idx = j
                    best = trial_denorm
            #................................................................
        
          
        yield best, cost[best_idx]

# ----------------------------------------------------------------------------

    
    
def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
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
            y += (w[i] * (x**i))
    
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(sum((Y_pred-Y)**2).mean())


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    '''
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000, verbose=False)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w < target_cost : # Added stop when current cost of best solution is less than target cost / Max
            break
    '''
        
    # Print the search result
    # print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000, verbose=False))    
    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = 0 #clf.score( 'INSERT MISSING CODE HERE'
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed =  'INSERT MISSING CODE HERE'


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    
    for i, p in enumerate(de_gen):
        w, c_w =    'INSERT MISSING CODE HERE'
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def task_3():
    '''
    Place holder for Task 3    
    '''
    pass
    'INSERT MISSING CODE HERE'


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    pass
    task_1()    
#    task_2()    
#    task_3()    

