import numpy as np
import pandas as pd
from typing import Callable


def compute_loss_mse(y, tx, w):
    """Computes the loss using MSE.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # compute error e
    e = y - np.dot(tx, w)
    # compute loss
    loss = 1 / 2 * np.mean(e**2)
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

    e = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, e) / len(e)
    return gradient


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    e = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, e) / len(e)
    return gradient, e


def sigmoid(x):
    """Applies sigmoid function on x.

    Args:
        x: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    return 1.0 / (1 + np.exp(-x))


def compute_loss_logistic(y, tx, w):
    """Computes the loss using logistic regression.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    p = sigmoid(np.dot(tx, w))
    # compute loss
    loss = np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
    return loss


def compute_gradient_logistic(y, tx, w):
    """Computes the gradient at w using logistic regression.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    p = sigmoid(np.dot(tx, w))
    # compute gradient
    gradient = np.dot(tx.T, (p - y)) / len(y)
    return gradient


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        degree: integer.

    Returns:
        poly: numpy array of shape (N, degree + 1)
    """

    num_samples, num_features = x.shape
    poly = np.zeros((num_samples, num_features * (degree + 1)))

    for d in range(degree + 1):
        poly[:, num_features * d : num_features * (d + 1)] = x**d

    return poly


def drop_columns(x, threshold):
    """
    Drop features with missing values above a certain threshold.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        threshold: float between 0 and 1.

    Returns:
        x with dropped columns
    """

    nan_count = np.isnan(x).sum(axis=0)
    nan_ratio = nan_count / x.shape[0]
    columns_to_drop_indices = np.where(nan_ratio > threshold)[0]
    columns_to_drop = columns_to_drop_indices.tolist()
    all_columns = np.arange(x.shape[1])
    columns_to_keep = np.delete(all_columns, columns_to_drop)

    return x[:, columns_to_keep]


def median_imputation(x):
    """
    Impute missing values in x with the median of the column.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.

    Returns:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
    """
    x_fill = x.copy()

    median = np.nanmedian(x_fill, axis=0)

    nan_indices = np.isnan(x_fill)

    x_fill[nan_indices] = np.take(median, np.where(nan_indices)[1])

    return x_fill


def mean_imputation(x):
    """
    Impute missing values in x with the median of the column.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.

    Returns:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
    """
    # Calculate the median of each column ignoring NaN values
    x_fill = x.copy()

    mean = np.nanmean(x_fill, axis=0)

    nan_indices = np.isnan(x_fill)

    x_fill[nan_indices] = np.take(mean, np.where(nan_indices)[1])

    return x_fill


def standardize(x):
    """
    Standardizes the input data.

    Args:
    x (numpy.ndarray): The input data as a NumPy array.

    Returns:
    numpy.ndarray: The standardized data.
    """

    # find mean and standard deviation
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)

    # epsilon to avoid division by zero
    epsilon = 1e-8
    std_dev[std_dev < epsilon] = epsilon

    # standardize
    standardized = (x - mean) / std_dev

    return standardized


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """

    num_rows = y.shape[0]
    # get the interval of each fold
    interval = int(num_rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_rows)
    # get an array of indices
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
