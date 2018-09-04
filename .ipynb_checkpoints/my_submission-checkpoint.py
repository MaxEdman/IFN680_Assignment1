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
                           verbose = True): # For testing add verbose2 = True
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
    w = np.random.rand(popsize, n_dimensions)
    #print(population)
    
    # Denormalizes the parameters to the corresponding values.
    # Need further commenting explanations of these steps.
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    w_denorm = min_b + (w * diff)
    cost = np.asarray([fobj(i) for i in w_denorm])
    
    best_idx = np.argmin(cost)
    best = w_denorm[best_idx]
    
    if verbose:
        print(
        '** Lowest cost in initial population = {} '
        .format(cost[best_idx]))        
    for i in range(maxiter):
        #if verbose2: For testing
        if verbose:
            print('** Starting generation {}, '.format(i+1))    
        
        for k in range(popsize) :
            #................................................................
            # Defines a list of indexes in the population exluding the current vector
            idxs = [idx for idx in range(popsize) if idx != k]
            
            # Select by random 3 vectors in the population from the idxs above
            a, b, c = w[np.random.choice(idxs, 3, replace = False)]
            
            # Creates a mutant vector based on the 3 selected vectors and clips the entries to the interval [0,1]
            mut = np.clip(a + mut * (b - c), 0, 1)
            
            # Initiates an array containing booleans if the value at index i in the current vector should be    replaces by the respective value in the mutant vector. These booleans are based on the predictive value in crossp.
            change_value = np.random.rand(n_dimensions) < crossp
            
            # If all values returned are false then one randomised element in the array is set to true. Otherwise the trial vector would not be different from the original vector.
            if not np.any(change_value):
                change_value[np.random.randint(0, n_dimensions)] = True
            
            # Creates a trial vector where the values from the mutant vectors have been switched on the positions denoted by True values in the change_value vector.
            trial = np.where(change_value, mut, w[k])
            
            # Denormalises the trial vector in order to evaluate it with the provided cost function fobj.
            trial_denorm = min_b + (trial * diff)
            f = fobj(trial_denorm)

            # If the cost of the trial vector is less than the cost of the original vector than this vector, and associated cost, are replaced. Unless the 
            if f < cost[k] and all(bounds[j][0]<=trial_denorm[j]<=bounds[j][1] for j in range(len(bounds))):
                cost[k] = f
                w[k] = trial
                
            # If the current cost is less than the previous best cost, then this index is replaced.
                if f < cost[best_idx] :
                    best_idx = k
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
        # To calculate the mean square error between the Y returned by fmodel and the initial Y values.
        return np.sqrt(sum((Y-Y_pred)**2) / len(Y)) 
        


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w < target_cost : # Added stop when current cost of best solution is less than target cost
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i+1,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()  
    
    ''' Configured for testing
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, maxiter=500, verbose=True, verbose2=False)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.3
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w < target_cost : # Added stop when current cost of best solution is less than target cost / Max
            break 
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    
    
    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=500, verbose=True, verbose2=False))
    w2 = result[-1][0]
    c_w2 = result[-1][1]
    
    print('Best cost found is {}'.format(c_w2))
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.plot(X, fmodel(X, w2), 'b-',label='model 2') # Added this to plot both usages of DE.
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()
    
    # Shows the lowest cost returned by fmodel for the different number of iterations of all vectors in the population.
    #x, f = zip(*result)
    #plt.plot(f)
    '''
    
    
    
    

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

        #verbose = 10 # Original value 
        verbose = False
        
        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=verbose, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        #mean_accuracy = 0 #clf.score( 'INSERT MISSING CODE HERE'
        # Sets the mean accuracy to the test score for the MLPClassifier.
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset/dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset/dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=False)
    
    for i, p in enumerate(de_gen):
        w, c_w = p
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i+1,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def task_3():
    '''
    The purpose of task_3 is to perform experiments by comparing the population size and maximum number of iterations. The results will show what combination gives the best results for training neural networks on a computational budget.
    
    The array to be tested is the following:
    x = [(5,40), (10,20),(20,10),(40,5)]
    Where every entry in x is a pair of (population size, max iterations).
    '''
    
    def test_computational_budget(population_size, max_iter):
        
        # For every pair create a new DE generator and transform it to a list. 
        result = list(differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=population_size, 
            maxiter=max_iter,
            verbose=True))
        
        # Zips the result into multiple arrays.
        x, f = zip(*result)
        return f
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    # This function needs to be included as a nested function in Task_3 as well.
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
        
        #Original value verbose = 10
        verbose = False

        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=verbose, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        #mean_accuracy = 0 #clf.score( 'INSERT MISSING CODE HERE'
        # Sets the mean accuracy to the test score for the MLPClassifier.
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return 1-mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset/dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset/dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters

    # Array containing pairs of population_size = 0, max_iter = 1.
    #x = [(5,40,'g'),(10,20,'b'),(20,10,'c'),(40,5,'y')]
    x = [(10,20,'b'),(20,10,'c'),(40,5,'y')]
    #x = [(4,2,'g'), (4,4,'b')]
    
    # Number of samples for each test to run.
    sample_size = 10
    
    # Colours to plot the different graphs.
    #colours = ['g','b','c','y']

    # Loop through the array of pairs to evaluate the results. population_size = 0, max_iter = 1.
    for pair in x :
        # Empty array to contain the sample data.
        dataArray = np.zeros(shape=(sample_size, pair[1]))
        
        # Loops n times to get data from the function defined in task_2.
        for i in range(sample_size):
            print('Starting sample round number {}'.format(i))
            dataArray[i] = test_computational_budget(pair[0], pair[1])
        
        # Calculates the mean data over the columns.
        meanData = dataArray.mean(axis=0)
        print("MeanData:")
        print(meanData)
        print("DataArray:")
        print(dataArray)
        
        # Calculates the standard deviaton to use as an error value.
        _STD = np.std(dataArray, axis=0) # Might change axis if its not returning correct values.
        print("Standard Deviation: {}".format(_STD))
        
        # Defines array of x-values.
        x_values = np.linspace(start=1, stop=pair[1], num=pair[1])
        
        # Creates a string as a label for the current graph. Pop_size = 0, max_iter = 1
        label_str = 'Pop_size {} - Max_iter {}'.format(pair[0], pair[1])
        
        # Plots the whole thingy
        plt.errorbar(x=x_values, y=meanData, yerr=_STD, c=pair[2], ecolor='r', fmt='o-', label=label_str)
        
        # One plot for each run.
        plt.xlabel('Number of Iterations')
        plt.ylabel('Error Score')
        plt.legend()
        plt.title('Comparing iterations and population size.')
        plt.show()
    

# ----------------------------------------------------------------------------









if __name__ == "__main__":
    pass
#   task_1()    
#   task_2()    
#   task_3()   