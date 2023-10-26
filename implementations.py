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


def cross_validation(
    y, x, k_indices, k, lambda_=0.1, max_iters=100, gamma=0.01, mod="ridge"
):
    """Cross validation for different models
    Args:
        y: numpy array of shape = (N, )
        x: numpy array of shape = (N, D)
        k_indices: numpy array of shape = (k, N/k)
        k: a scalar denoting the number of folds
        lambda_: a scalar denoting the regularization parameter
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the step size
        mod: a string denoting the model to use
    Returns:
        losses_train: numpy array of shape = (k, ) containing the training losses
        losses_test: numpy array of shape = (k, ) containing the testing losses
        accuracies_train: numpy array of shape = (k, ) containing the training accuracies
        accuracies_test: numpy array of shape = (k, ) containing the testing accuracies
        ws: numpy array of shape = (k, D) containing the weights
    """

    (
        losses_train,
        losses_test,
        accuracies_train,
        accuracies_test,
        f1s_train,
        f1s_test,
        ws,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for kth in range(k):
        test_indices = k_indices[kth]
        train_indices = k_indices[
            np.arange(k_indices.shape[0]) != kth
        ]  # all but kth element
        train_indices = train_indices.reshape(-1)

        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]

        initial_w = np.zeros(x.shape[1])

        if mod == "ridge":
            w, _ = ridge_regression(y_train, x_train, lambda_)

        elif mod == "logistic":
            w, _ = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)

        elif mod == "reg_logistic":
            w, _ = reg_logistic_regression(
                y_train, x_train, initial_w, max_iters, gamma, lambda_
            )

        elif mod == "least_squares":
            w, _ = least_squares(y_train, x_train)

        elif mod == "mse_gd":
            w, _ = mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)

        elif mod == "mse_sgd":
            w, _ = mean_squared_error_sgd(y_train, x_train, initial_w, max_iters, gamma)

        else:
            raise ValueError("Unknown model")

        ws.append(w)

        pred_train = np.dot(x_train, w)
        pred_test = np.dot(x_test, w)

        if mod == "logistic" or mod == "reg_logistic":
            pred_train = sigmoid(pred_train)
            pred_test = sigmoid(pred_test)

        prediction_train = np.where(pred_train >= 0.5, 1, 0)
        prediction_test = np.where(pred_test >= 0.5, 1, 0)

        loss_train = compute_loss_mse(y_train, x_train, w)
        loss_test = compute_loss_mse(y_test, x_test, w)
        losses_train.append(np.sqrt(2 * loss_train))
        losses_test.append(np.sqrt(2 * loss_test))

        accuracy_train = np.mean(y_train == prediction_train)
        accuracy_test = np.mean(y_test == prediction_test)
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)

        f1_train = f1_score(y_train, prediction_train)
        f1_test = f1_score(y_test, prediction_test)
        f1s_train.append(f1_train)
        f1s_test.append(f1_test)

        print("Cross validation: {kth}/{k}".format(kth=kth + 1, k=k))
        print("Training loss: {l}".format(l=loss_train))
        print("Testing loss: {l}".format(l=loss_test))
        # print("Training accuracy: {a}".format(a=accuracy_train))
        # print("Testing accuracy: {a}".format(a=accuracy_test))
        print("Training f1 score: {f}".format(f=f1_train))
        print("Testing f1 score: {f}".format(f=f1_test))
        print("")

    return (
        np.mean(losses_train),
        np.mean(losses_test),
        np.mean(accuracies_train),
        np.mean(accuracies_test),
        np.mean(f1s_train),
        np.mean(f1s_test),
        np.mean(ws, axis=0),
    )
