import numpy as np
import pandas as pd


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


def columns_to_drop_numpy(x, threshold):
    """
    Drop features with missing values above a certain threshold.
    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        threshold: float between 0 and 1.
    Returns:
        columns_to_drop: numpy array of shape (D', )
    """

    nan_count = np.isnan(x).sum(axis=0)
    nan_ratio = nan_count / x.shape[0]
    columns_to_drop_indices = np.where(nan_ratio > threshold)[0]
    columns_to_drop = columns_to_drop_indices.tolist()

    return columns_to_drop


def median_imputation_numpy(x):
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


def standardize_data(x):
    """
    Standardizes the input data.

    Parameters:
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
