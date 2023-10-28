""" This file contains some helper functions for data cleaning and feature processing. """
import numpy as np


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

    return x[:, columns_to_keep], columns_to_keep


def drop_correlated_columns(data, threshold):
    """
    Drop correlated columns from a numpy matrix based on a correlation threshold.

    Parameters:
    - data (numpy array): The data matrix.
    - threshold (float): The correlation threshold.

    Returns:
    - The data matrix after dropping correlated columns.
    """
    corr_matrix = np.corrcoef(data, rowvar=False)
    upper_triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlated_pairs = np.column_stack(
        np.where(abs(corr_matrix * upper_triangle_mask) > threshold)
    )
    columns_to_drop = set()
    for i, j in correlated_pairs:
        columns_to_drop.add(j)

    all_columns = np.arange(data.shape[1])
    columns_to_drop = np.array(
        list(columns_to_drop), dtype=int
    )  # Convert set to numpy array
    columns_to_keep = np.delete(all_columns, columns_to_drop)

    reduced_data = np.delete(data, columns_to_drop, axis=1)

    return reduced_data, columns_to_keep


def drop_single_value_columns(data):
    """
    Drop columns with a single unique value from a numpy array.

    Parameters:
    - data (numpy array): The data matrix.

    Returns:
    - The data matrix after dropping the columns.
    """
    columns_to_drop = []
    for i in range(data.shape[1]):
        if len(np.unique(data[:, i])) == 1:
            columns_to_drop.append(i)

    all_columns = np.arange(data.shape[1])
    columns_to_drop = np.array(
        list(columns_to_drop), dtype=int
    )  # Convert set to numpy array
    columns_to_keep = np.delete(all_columns, columns_to_drop)
    reduced_data = np.delete(data, columns_to_drop, axis=1)

    return reduced_data, columns_to_keep


def remove_outliers(x, low_perc, high_perc):
    """
    Remove outliers
    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        low_perc: numpy array of shape (D,) containing the lower percentile for each feature.
        high_perc: numpy array of shape (D,) containing the upper percentile for each feature.
    Returns:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
    """
    for i in range(np.shape(x)[1]):
        x[:, i][x[:, i] < low_perc[i]] = low_perc[i]
        x[:, i][x[:, i] > high_perc[i]] = high_perc[i]
    return x


def median_imputation(x):
    """Impute missing values in x with the median of the column.

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
    """Impute missing values in x with the median of the column.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.

    Returns:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
    """
    x_fill = x.copy()
    mean = np.nanmean(x_fill, axis=0)
    nan_indices = np.isnan(x_fill)
    x_fill[nan_indices] = np.take(mean, np.where(nan_indices)[1])
    return x_fill


def standardize(x):
    """Standardizes the input data.

    Args:
        x (numpy.ndarray): The input data as a NumPy array.

    Returns:
        standardized (numpy.ndarray): The standardized data.
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


def build_poly(x, degree):
    """Polynomial basis functions for input data x.

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


def build_log(x, epsilon=1e-10):
    """Logarithm basis functions for input data x. Stacks original features and log features."""
    log_transformation = np.log(np.abs(x) + epsilon)

    return log_transformation


def build_ratios(x, epsilon=1e-10):
    """Calculate the ratio of each feature compared to its average, maximum, and minimum.

    Args:
        x: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        epsilon: small constant to prevent division by zero.

    Returns:
        ratio_features: numpy array of shape (N, 3D) containing the new features.
    """

    # calculate average, max, and min across all features for each sample
    avg_features = np.mean(x, axis=1, keepdims=True)
    min_features = np.min(x, axis=1, keepdims=True)
    max_features = np.max(x, axis=1, keepdims=True)
    # calculate ratios
    ratio_to_avg = x / (avg_features + epsilon)
    ratio_to_min = x / (min_features + epsilon)
    ratio_to_max = x / (max_features + epsilon)

    # stack all ratios
    ratio_features = np.hstack((ratio_to_avg, ratio_to_min, ratio_to_max))

    return ratio_features


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold.

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
