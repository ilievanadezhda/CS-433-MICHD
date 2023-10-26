import numpy as np
from implementations_utils import sigmoid


# Prediction functions
def predict_mse(tx, w):
    return np.where(np.dot(tx, w) >= 0.5, 1, 0)


def predict_logistic(tx, w):
    return np.where(sigmoid(np.dot(tx, w)) >= 0.5, 1, 0)


# Evaluation metrics
def accuracy(y_true, y_pred):
    """Computes the accuracy.
    Args:
        y_true: numpy array of shape = (N, ). The true labels.
        y_pred: numpy array of shape = (N, ). The predicted labels.
    Returns:
        The accuracy (a scalar)
    """
    return np.mean(y_true == y_pred)


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
    return np.mean(np.nan_to_num(f1))


# Print functions
def print_fold_results(
    idx, k, eval_results_train, eval_results_test, losses_train, losses_test
):
    print("Fold {idx}/{k}".format(idx=idx + 1, k=k))
    print("Train loss: {l:.5f}".format(l=losses_train[idx]))
    print("Test loss: {l:.5f}".format(l=losses_test[idx]))
    for key, value in eval_results_train.items():
        print("Train {key}: {value:.5f}".format(key=key, value=value[idx]))
    for key, value in eval_results_test.items():
        print("Test {key}: {value:.5f}".format(key=key, value=value[idx]))
    print("-" * 30)


def print_results(eval_results):
    means = {key: np.mean(values) for key, values in eval_results.items()}
    stds = {key: np.std(values) for key, values in eval_results.items()}
    for key in eval_results.keys():
        print(f"{key} : {means[key]:.6f} ± {stds[key]:.6f}")


# Cross validation
def cross_validation(y, tx, k_indices, model_fn, loss_fn, pred_fn, eval_fns=dict()):
    k = len(k_indices)
    all_weights = []
    losses_train, losses_test = [], []
    eval_results_train, eval_results_test = {key: [] for key in eval_fns.keys()}, {
        key: [] for key in eval_fns.keys()
    }
    for idx in range(k):
        # get test/train indices
        test_idx, train_idx = k_indices[idx], k_indices[np.arange(k) != idx].reshape(-1)
        # split the data in train/test set
        x_train, y_train = tx[train_idx], y[train_idx]
        x_test, y_test = tx[test_idx], y[test_idx]
        # initialize weights
        initial_w = np.zeros(x_train.shape[1])
        # train model
        trained_w, _ = model_fn(y_train, x_train, initial_w)
        # store weights
        all_weights.append(trained_w)
        # compute loss
        losses_train.append(loss_fn(y_train, x_train, trained_w))
        losses_test.append(loss_fn(y_test, x_test, trained_w))
        # compute predictions
        y_pred_train = pred_fn(x_train, trained_w)
        y_pred_test = pred_fn(x_test, trained_w)
        # compute evaluation metrics
        for key, value in eval_fns.items():
            # append train
            eval_results_train[key].append(value(y_train, y_pred_train))
            # append test
            eval_results_test[key].append(value(y_test, y_pred_test))
        # print results
        print_fold_results(
            idx, k, eval_results_train, eval_results_test, losses_train, losses_test
        )
    return (
        eval_results_train,
        eval_results_test,
        losses_train,
        losses_test,
        np.mean(all_weights, axis=0),
    )
