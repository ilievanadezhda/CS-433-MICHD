import numpy as np
from helpers import batch_iter
from implementations_utils import (
    compute_loss_logistic,
    compute_gradient_logistic,
    compute_loss_mse,
    compute_gradient_mse,
    compute_stoch_gradient,
    sigmoid,
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
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update weights
            w = w - gamma * grad
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


def cross_validation(y, x, k_indices, k, model_func, model_args={}):
    """Return the loss of the model for k folds.

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, number of folds
        model_func: function to compute the weights and loss
        model_args: dict, arguments to pass to the model function

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse) and the weights w
    """
    losses_train, losses_test, accuracies_train, accuracies_test, ws = (
        [],
        [],
        [],
        [],
        [],
    )

    for kth in range(k):
        print("kth={}".format(kth))
        test_indices = k_indices[kth]
        train_indices = k_indices[
            np.arange(k_indices.shape[0]) != kth
        ]  # all but kth element
        train_indices = train_indices.reshape(-1)

        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]

        w, final_loss = model_func(y_train, x_train, **model_args)
        ws.append(w)

        loss_train = compute_loss_mse(
            y_train, x_train, w
        )  # Modify this if you have a different loss for a specific model
        loss_test = compute_loss_mse(y_test, x_test, w)  # Modify this if needed

        # accuracy for train and test
        pred_probs_train = sigmoid(np.dot(x_train, w))
        predictions_train = np.where(pred_probs_train >= 0.5, 1, 0)

        pred_probs_test = sigmoid(np.dot(x_test, w))
        predictions_test = np.where(pred_probs_test >= 0.5, 1, 0)

        accuracy_train = np.mean(y_train == predictions_train)
        accuracy_test = np.mean(y_test == predictions_test)
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)

        losses_train.append(loss_train)
        losses_test.append(loss_test)
        # print accuracies
        print(
            "fold={k}, accuracy_train={l_tr:.3f}, accuracy_test={l_te:.3f},".format(
                k=kth, l_tr=accuracies_train[-1], l_te=accuracies_test[-1]
            )
        )
        print(
            "loss_train={l_tr:.3f}, loss_test={l_te:.3f},".format(
                l_tr=losses_train[-1], l_te=losses_test[-1]
            )
        )

    return (
        np.mean(losses_train),
        np.mean(losses_test),
        np.mean(accuracies_train),
        np.mean(accuracies_test),
        np.mean(ws, axis=0),
    )
