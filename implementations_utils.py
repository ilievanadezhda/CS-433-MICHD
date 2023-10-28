""" This file contains some helper functions used in implementations.py."""
import numpy as np


def compute_loss_mse(y, tx, w):
    """Computes the loss using MSE.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters w.
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
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, e) / len(e)
    return gradient, e


def sigmoid(x):
    """Applies sigmoid function on x.

    Args:
        x: scalar or numpy array

    Returns:
        Scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-x))


def compute_loss_logistic(y, tx, w):
    """Computes the loss using logistic regression.

    Args:
        y: numpy array of shape = (N, )
        tx: numpy array of shape = (N, D)
        w: numpy array of shape = (D, ). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters w.
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


def f1_score(y_true, y_pred):
    """Computes the F1 score.
    Args:
        y_true: numpy array of shape = (N, ). The true labels.
        y_pred: numpy array of shape = (N, ). The predicted labels.
    Returns:
        The F1 score (a scalar)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(y_true)

    # If it's binary, just consider one of the classes as positive
    if len(classes) == 2:
        classes = np.array([1])

    tp = np.zeros_like(classes, dtype=np.float64)
    fp = np.zeros_like(classes, dtype=np.float64)
    fn = np.zeros_like(classes, dtype=np.float64)

    for i, c in enumerate(classes):
        tp[i] = np.sum((y_true == c) & (y_pred == c))
        fp[i] = np.sum((y_true != c) & (y_pred == c))
        fn[i] = np.sum((y_true == c) & (y_pred != c))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)

    f1_score = np.mean(f1)

    return f1_score
