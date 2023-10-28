"""This file contains a pipeline for data preprocessing and model evaluation on logistic regression and regularized logistic regression. """
import numpy as np
from feature_processing import (
    drop_columns,
    median_imputation,
    mean_imputation,
    drop_single_value_columns,
    drop_correlated_columns,
    build_poly,
    build_log,
    build_ratios,
    build_k_indices,
    standardize,
)
from cross_validation import (
    predict_logistic,
    accuracy,
    f1_score,
    print_results,
    cross_validation,
)
from implementations import (
    logistic_regression,
    reg_logistic_regression,
)
from implementations import (
    compute_loss_logistic,
)


def execute_pipeline(
    x_train,
    x_test,
    y_train,
    DROP_NAN_THRESHOLD,
    DROP_CORR_THRESHOLD,
    CAT_NUM_THRESHOLD,
    DROP_SINGLE,
    BUILD_RATIOS,
    BUILD_LOG,
    BUILD_POLY,
    DEGREE,
    STANDARDIZE,
    NUM_FOLDS,
    GAMMA_,
    MAX_ITERS_,
    LAMBDA,
    results,
):
    x_train_temp, cols_to_keep_1 = drop_columns(x_train, DROP_NAN_THRESHOLD)
    x_test_temp = x_test[:, cols_to_keep_1]
    print(f"Dropping columns with DROP_NAN_THRESHOLD = {DROP_NAN_THRESHOLD}...")

    # Identify categorical and numerical features
    categorical_features = []
    numerical_features = []
    for i, feature in enumerate(x_train_temp.T):
        if np.unique(feature).shape[0] < CAT_NUM_THRESHOLD:
            categorical_features.append(i)
        else:
            numerical_features.append(i)

    # Impute missing values on train and test datasets
    x_train_temp[:, categorical_features] = median_imputation(
        x_train_temp[:, categorical_features]
    )
    x_test_temp[:, categorical_features] = median_imputation(
        x_test_temp[:, categorical_features]
    )
    x_train_temp[:, numerical_features] = mean_imputation(
        x_train_temp[:, numerical_features]
    )
    x_test_temp[:, numerical_features] = mean_imputation(
        x_test_temp[:, numerical_features]
    )

    # Data engineering: ratios, log-transform, polynomial features
    if BUILD_RATIOS:
        print("Building ratios...")
        ratio_features_train = build_ratios(x_train_temp[:, numerical_features])
        ratio_features_test = build_ratios(x_test_temp[:, numerical_features])
        x_train_temp = np.hstack((x_train_temp, ratio_features_train))
        x_test_temp = np.hstack((x_test_temp, ratio_features_test))

    if BUILD_LOG:
        print("Building log...")
        log_features_train = build_log(x_train_temp[:, numerical_features])
        log_features_test = build_log(x_test_temp[:, numerical_features])
        x_train_temp = np.hstack((x_train_temp, log_features_train))
        x_test_temp = np.hstack((x_test_temp, log_features_test))

    if BUILD_POLY:
        print(f"Building polynomial with degree = {DEGREE}...")
        x_train_temp = build_poly(x_train_temp, DEGREE)
        x_test_temp = build_poly(x_test_temp, DEGREE)

    # Drop columns with a single unique value or highly correlated columns
    if DROP_SINGLE:
        x_train_temp, cols_to_keep_3 = drop_single_value_columns(x_train_temp)
        x_test_temp = x_test_temp[:, cols_to_keep_3]
        print("Dropping single valued columns...")

    x_train_temp, cols_to_keep_2 = drop_correlated_columns(
        x_train_temp, DROP_CORR_THRESHOLD
    )
    x_test_temp = x_test_temp[:, cols_to_keep_2]

    # Standardize the datasets
    if STANDARDIZE:
        print("Standardizing...")
        x_train_temp = standardize(x_train_temp)
        x_test_temp = standardize(x_test_temp)

    # Cross-validation and model evaluation
    k_indices = build_k_indices(y_train, NUM_FOLDS, 42)
    MAX_ITERS = MAX_ITERS_
    GAMMA = GAMMA_
    INITIAL_W = np.zeros(x_train_temp.shape[1])

    # Model

    def _logistic_regression(y, tx, initial_w):
        return logistic_regression(y, tx, INITIAL_W, MAX_ITERS, GAMMA)

    # Cross validation with logistic and reg logistic regression
    (
        eval_results_train,
        eval_results_test,
        losses_train,
        losses_test,
        w,
    ) = cross_validation(
        y_train,
        x_train_temp,
        k_indices,
        model_fn=_logistic_regression,
        loss_fn=compute_loss_logistic,
        pred_fn=predict_logistic,
        eval_fns=dict(accuracy=accuracy, f1_score=f1_score),
    )

    logistic_regression_result = {
        "Drop Threshold": DROP_NAN_THRESHOLD,
        "Drop corr thresh.": DROP_CORR_THRESHOLD,
        "Imputation": CAT_NUM_THRESHOLD,
        "Standardization": STANDARDIZE,
        "Build Poly": BUILD_POLY,
        "Degree": DEGREE,
        "Build Log": BUILD_LOG,
        "Build Ratios": BUILD_RATIOS,
        "Model": "Logistic_Regression",
        "Initial W": INITIAL_W,
        "Max Iters": MAX_ITERS,
        "Gamma": GAMMA,
        "Lambda": -999,  # Not used
        "CV F1 std": np.std(eval_results_test["f1_score"]),
        "CV Accuracy std": np.std(eval_results_test["accuracy"]),
        "CV F1": np.mean(eval_results_test["f1_score"]),
        "CV Accuracy": np.mean(eval_results_test["accuracy"]),
    }
    results.append(logistic_regression_result)

    print("Train:")
    print_results(eval_results_train)
    print("-" * 20)
    print("Test:")
    print_results(eval_results_test)
    ### Regularized logistic regression
    # Model parameters
    MAX_ITERS = MAX_ITERS_
    GAMMA = GAMMA_
    LAMBDA_ = LAMBDA

    def _reg_logistic_regression(y, tx, initial_w):
        return reg_logistic_regression(y, tx, LAMBDA_, initial_w, MAX_ITERS, GAMMA)

    # Cross validation
    (
        eval_results_train,
        eval_results_test,
        losses_train,
        losses_test,
        w,
    ) = cross_validation(
        y_train,
        x_train_temp,
        k_indices,
        model_fn=_reg_logistic_regression,
        loss_fn=compute_loss_logistic,
        pred_fn=predict_logistic,
        eval_fns=dict(accuracy=accuracy, f1_score=f1_score),
    )

    logistic_regression_result = {
        "Drop Threshold": DROP_NAN_THRESHOLD,
        "Drop corr thresh.": DROP_CORR_THRESHOLD,
        "Imputation": CAT_NUM_THRESHOLD,
        "Standardization": STANDARDIZE,
        "Build Poly": BUILD_POLY,
        "Degree": DEGREE,
        "Build Log": BUILD_LOG,
        "Build Ratios": BUILD_RATIOS,
        "Model": "Reg_Logistic_Regression",
        "Initial W": INITIAL_W,
        "Max Iters": MAX_ITERS,
        "Gamma": GAMMA,
        "Lambda": LAMBDA_,
        "CV F1 std": np.std(eval_results_test["f1_score"]),
        "CV Accuracy std": np.std(eval_results_test["accuracy"]),
        "CV F1": np.mean(eval_results_test["f1_score"]),
        "CV Accuracy": np.mean(eval_results_test["accuracy"]),
    }
    results.append(logistic_regression_result)

    print("Train:")
    print_results(eval_results_train)
    print("-" * 20)
    print("Test:")
    print_results(eval_results_test)
    return results
