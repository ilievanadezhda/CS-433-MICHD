import numpy as np

""" Helper functions for mean_squared_error_gd"""
def compute_loss_mse(y, tx, w):
    """Computes the loss using MSE.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # obtain N
    N = y.shape[0]
    # compute error e
    e = y - np.matmul(tx, w)
    # compute loss
    loss = 1/(2*N) * np.matmul(e.T, e)
    return loss

def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    # obtain N
    N = y.shape[0]
    # compute e
    e = y - np.matmul(tx, w)
    # compute gradient
    gradient = - 1/N * np.matmul(tx.T, e)
    return gradient
