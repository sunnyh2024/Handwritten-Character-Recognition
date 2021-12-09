from sklearn.utils import shuffle
import numpy as np

# Note: This was created at the beginning before we realized on how Keras worked
# and we didn't realize how difficult it is to implement it for our purposes.
# As a result, this is not used at all.
# We tried looking at the documentation for Keras and thats how we ended up with SGDv2.py

def SGD(x, y, theta_init, lr, tolerance, maxIter):
    '''
    This SGD computes gradient values using each point of training data, 
    update weight values in the opposite direction of the gradient of the objective 
    function and compute the value of loss function respect to updated weights. 
    SGD algorithm will end before maxIter if the relative difference between
    the current weight and the previous weight is 
    less than covergence tolerance (tolerance).
    
    Arguments:
    x             -- training data for SGD  
    y             -- training labels for SGD.
    theta_init    -- initial weights.
    lr            -- learning rate.
    tolerance     -- convergence tolerance.
    maxIter       -- number of iterations that SGD should be run
    
    Return:
    thetas        -- A list of weight values, saving for each iteration.
    GDs           -- A list of gradient values, saving for each iteration.
    losses        -- A list of values of the loss function, saving for each iteration.
    '''
  
    # X is extended input matrix by adding "1" column
    X = np.vstack((np.ones(len(x)),x.T))
    # xy is training data matrix (combine x and y)
    xy = np.vstack((x,y)).T
    
    # history vectors
    thetas = [theta_init]
    oldtheta = theta_init
    GDs = []
    losses = [computeloss(X, y, theta_init)]
    
    for _ in range(maxIter):

        # shuffe training data after each epoch
        np.random.shuffle(xy)
        # accumulate gradient values
        accGrad = 0
        
        # xs, ys are input and output data sample (data point) in training data
        for xi, yi in xy:
            Xi= np.array([1, xi])
            # current 'updated' gradient values (newGrad)
            Grad = computeGradient(Xi,yi,oldtheta)
            # updated weights (newtheta)
            theta = updateWeights(oldtheta,Grad,lr)
            accGrad += Grad
            oldtheta = theta

        thetas.append(theta)
        # gradient value after each epoch is the mean of gradient value at each data point
        GDs.append(accGrad/len(x))
        losses.append(computeloss(X, y, thetas[-1]))

        # thetas[-1]: current weight, thetas[-2]: previous weight
        if isConverged(thetas[-1], thetas[-2], tolerance):
            break
    return np.array(thetas), np.array(GDs), np.array(losses)


def computeGradient(X,y,theta):
    '''
    Compute the gradient values based on formula Grad = X*(X.T*theta - y) 
    '''
    Grad = X.dot(X.T.dot(theta)-y)/y.size
    return Grad

def updateWeights(w, Grad, eta):
    '''
    Update weights based on formula w = w - eta*Grad 
    '''  
    w = w - eta*Grad
    return w

def computeloss(X, y, theta):
    '''
    Compute the value of loss function given updated weight with the formula loss = (X.T*theta - y)^2
    '''
    temp = np.dot(X.T, theta) - y
    loss = 0.5*np.dot(temp, temp.T)/y.size
    return loss

def isConverged(w1, w2, tolerance):
    '''
    Check whether or not convergence is reached by checking the relative
    difference between the current weight and the previous weight is less
    than convergence tolerance value. 
    '''
    RelDiff = np.linalg.norm(w1 - w2)/len(w1)
    return (RelDiff < tolerance)