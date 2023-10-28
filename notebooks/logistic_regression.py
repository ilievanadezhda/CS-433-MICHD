"""This file contains the random search for the best hyperparameters for logistic regression."""

import sys

sys.path.append("../")
# add ../ to path
import csv
import random
import numpy as np
from helpers import load_csv_data
from feature_processing import (
    drop_columns,
    drop_correlated_columns,
    drop_single_value_columns,
    median_imputation,
    mean_imputation,
    standardize,
    build_poly,
    build_k_indices,
    build_log,
    build_ratios,
)
from cross_validation import (
    predict_logistic,
    accuracy,
    f1_score,
    cross_validation,
)
from implementations import logistic_regression
from implementations_utils import compute_loss_logistic


def pipeline(
    DROP_NAN_THRESHOLD,
    CAT_NUM_THRESHOLD,
    DROP_CORR_THRESHOLD,
    BUILD_POLY,
    BUILD_LOG,
    BUILD_RATIOS,
    MAX_ITERS,
    GAMMA,
    k_indices,
    x_train,
    y_train,
):
    # drop_columns
    x_train, cols_to_keep_1 = drop_columns(x_train, DROP_NAN_THRESHOLD)
    # categorical and numerical features
    categorical_features = []
    numerical_features = []
    # find categorical and numerical features
    for i, feature in enumerate(x_train.T):
        if np.unique(feature).shape[0] < CAT_NUM_THRESHOLD:
            categorical_features.append(i)
        else:
            numerical_features.append(i)
    # fill in missing values on the train and test
    x_train[:, categorical_features] = median_imputation(
        x_train[:, categorical_features]
    )
    x_train[:, numerical_features] = mean_imputation(x_train[:, numerical_features])
    # drop_single_value_columns on x_train and x_test
    x_train, cols_to_keep_3 = drop_single_value_columns(x_train)
    # drop_correlated_columns on x_train and x_test
    x_train, cols_to_keep_2 = drop_correlated_columns(x_train, DROP_CORR_THRESHOLD)
    if BUILD_POLY:
        # build poly on x_train and x_test
        x_train = build_poly(x_train, 2)
    if BUILD_LOG:
        # build_log on x_train and x_test
        x_train = build_log(x_train)
    if BUILD_RATIOS:
        # build ratios between columns
        x_train = build_ratios(x_train)
    x_train = standardize(x_train)

    # define model function
    def _logistic_regression(y, tx, initial_w):
        return logistic_regression(y, tx, initial_w, MAX_ITERS, GAMMA)

    # cross validation
    (
        eval_results_train,
        eval_results_test,
        losses_train,
        losses_test,
        w,
    ) = cross_validation(
        y_train,
        x_train,
        k_indices,
        model_fn=_logistic_regression,
        loss_fn=compute_loss_logistic,
        pred_fn=predict_logistic,
        eval_fns=dict(accuracy=accuracy, f1_score=f1_score),
    )
    # return shape of x_train, eval_results_train, eval_results_test
    return x_train.shape, eval_results_train, eval_results_test


# load data
print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("../../data/")
print("Data loaded!")
# replace -1 with 0 in y_train
y_train[np.where(y_train == -1)] = 0
# number of folds
NUM_FOLDS = 5
# build k_indices for cross validation
k_indices = build_k_indices(y_train, NUM_FOLDS, 42)
# number of random combinations to try
combinations = 500
with open("logistic_regression.csv", "a", newline="") as csvfile:
    fieldnames = [
        "x_train_shape",
        "DROP_NAN_THRESHOLD",
        "CAT_NUM_THRESHOLD",
        "DROP_CORR_THRESHOLD",
        "BUILD_POLY",
        "BUILD_LOG",
        "BUILD_RATIOS",
        "MAX_ITERS",
        "GAMMA",
        "TRAIN ACCURACY",
        "TRAIN F1",
        "TEST ACCURACY",
        "TEST F1",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    for _ in range(combinations):
        # random hyperparameters
        DROP_NAN_THRESHOLD = random.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        CAT_NUM_THRESHOLD = random.choice(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
        )
        DROP_CORR_THRESHOLD = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        BUILD_POLY = random.choice([True, False])
        BUILD_LOG = random.choice([True, False])
        BUILD_RATIOS = random.choice([True, False])
        MAX_ITERS = random.choice([100, 200, 300, 400, 500])
        GAMMA = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        # call pipleline
        shape, eval_results_train, eval_results_test = pipeline(
            DROP_NAN_THRESHOLD,
            CAT_NUM_THRESHOLD,
            DROP_CORR_THRESHOLD,
            BUILD_POLY,
            BUILD_LOG,
            BUILD_RATIOS,
            int(MAX_ITERS),
            GAMMA,
            k_indices,
            x_train,
            y_train,
        )
        # write to csv
        writer.writerow(
            {
                "x_train_shape": shape,
                "DROP_NAN_THRESHOLD": DROP_NAN_THRESHOLD,
                "CAT_NUM_THRESHOLD": CAT_NUM_THRESHOLD,
                "DROP_CORR_THRESHOLD": DROP_CORR_THRESHOLD,
                "BUILD_POLY": BUILD_POLY,
                "BUILD_LOG": BUILD_LOG,
                "BUILD_RATIOS": BUILD_RATIOS,
                "MAX_ITERS": MAX_ITERS,
                "GAMMA": GAMMA,
                "TRAIN ACCURACY": f"{round(np.mean(eval_results_train['accuracy']), 6)} ± {round(np.std(eval_results_train['accuracy']), 6)}",
                "TRAIN F1": f"{round(np.mean(eval_results_train['f1_score']), 6)} ± {round(np.std(eval_results_train['f1_score']), 6)}",
                "TEST ACCURACY": f"{round(np.mean(eval_results_test['accuracy']), 6)} ± {round(np.std(eval_results_test['accuracy']), 6)}",
                "TEST F1": f"{round(np.mean(eval_results_test['f1_score']), 6)} ± {round(np.std(eval_results_test['f1_score']), 6)}",
            }
        )
        csvfile.flush()
        sys.stdout.flush()
