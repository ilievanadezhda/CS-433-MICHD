"""This file contains implementations of the required methods."""
import numpy as np
from helpers import batch_iter
from implementations_utils import (
    compute_loss_logistic,
    compute_gradient_logistic,
    compute_loss_mse,
    compute_gradient_mse,
    compute_stoch_gradient,
    sigmoid,
    f1_score,
)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        initial_w: numpy array of shape = (D, )
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the step size

    Returns:
        w: numpy array of shape = (D, ) corresponding to the last weight vector of the method
        loss: the corresponding loss value
    """
    # initialize w
    w = initial_w
    # initialize loss
    loss = compute_loss_mse(y, tx, w)
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_mse(y, tx, w)
        # update weights
        w = w - gamma * gradient
        # compute loss
        loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    # initialize w
    w = initial_w
    # initialize loss
    for n_iter in range(max_iters):
        # iterate over batches
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute gradient
            gradient, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update weights
            w = w - gamma * gradient
            # compute loss
            loss = compute_loss_mse(y, tx, w)

        # print(
        #     "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    X = np.dot(tx.T, tx)
    Y = np.dot(tx.T, y)
    w = np.linalg.solve(X, Y)
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    X = np.dot(tx.T, tx) + reg
    Y = np.dot(tx.T, y)
    w = np.linalg.solve(X, Y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y in {0,1})"""
    # initialize w
    w = initial_w
    # initialize loss
    loss = compute_loss_logistic(y, tx, w)
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_logistic(y, tx, w)
        # update weights
        w = w - gamma * gradient
        # compute loss
        loss = compute_loss_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD (y in {0,1},
    with regularization term lambda*|w|^2)
    """
    # initialize w
    w = initial_w
    # initialize loss
    loss = compute_loss_logistic(y, tx, w)
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        # update weights
        w = w - gamma * gradient
        # compute loss
        loss = compute_loss_logistic(y, tx, w)
    return w, loss
