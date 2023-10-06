from utils import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        initial_w: numpy array of shape = (D, )
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the step size 

    Returns:
        w: the last weight vector of the method
        loss: the corresponding loss value 
    """
    # initialize w
    w = initial_w
    # initialize loss
    loss = compute_loss_mse(y, tx, initial_w)
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_mse(y, tx, w)
        # compute loss
        loss = compute_loss_mse(y, tx, w)
        # update weights 
        w = w - gamma*gradient
    return (w, loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    """
    raise NotImplementedError

def least_squares(y, tx):
    """ Least squares regression using normal equations
    """
    raise NotImplementedError

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    """
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD (y in {0,1})
    """
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD (y in {0,1}, 
    with regularization term lambda*|w|^2)
    """
    raise NotImplementedError


