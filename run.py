"""This file is used to generate the best submission."""
import numpy as np
from helpers import load_csv_data, create_csv_submission
from feature_processing import (
    drop_columns,
    drop_correlated_columns,
    drop_single_value_columns,
    median_imputation,
    mean_imputation,
    standardize,
    build_poly,
    build_log,
    build_ratios,
    remove_outliers,
)
from cross_validation import f1_score
from neural_network import (
    Fully_connected,
    ReLU,
    train,
    predict,
    iterate_minibatches,
)

np.random.seed(0)

CONFIG = {
    "DROP_NAN_THRESHOLD": 1,
    "CAT_NUM_THRESHOLD": 200,
    "DROP_SINGLE": True,
    "DROP_CORR_THRESHOLD": 1,
    "REMOVE_OUTLIERS": False,
    "BUILD_POLY": False,
    "DEGREE": 2,
    "BUILD_LOG": False,
    "BUILD_RATIOS": False,
    "STANDARDIZE": True,
    "SPLIT_RATIO": 1,
}


def load_and_split_data(path):
    x_train, x_test, Y_train, train_ids, test_ids = load_csv_data(path)
    split_index = int(len(x_train) * CONFIG["SPLIT_RATIO"])
    X_train = x_train[:split_index]
    y_train = Y_train[:split_index]
    y_train[np.where(y_train == -1)] = 0
    return X_train, x_test, y_train, train_ids, test_ids


def preprocess_data(X_train, x_test):
    x_train_temp = X_train.copy()
    x_test_temp = x_test.copy()
    x_train_temp, cols_to_keep_1 = drop_columns(
        x_train_temp, CONFIG["DROP_NAN_THRESHOLD"]
    )
    x_test_temp = x_test_temp[:, cols_to_keep_1]
    print(
        f"Dropping columns with DROP_NAN_THRESHOLD = {CONFIG['DROP_NAN_THRESHOLD']}..."
    )

    # Identify categorical and numerical features
    # categorical and numerical features
    categorical_features = []
    numerical_features = []
    # find categorical and numerical features
    for i, feature in enumerate(x_train_temp.T):
        if np.unique(feature).shape[0] < CONFIG["CAT_NUM_THRESHOLD"]:
            categorical_features.append(i)
        else:
            numerical_features.append(i)
    # fill in missing values on the train and test
    x_train_temp[:, categorical_features] = median_imputation(
        x_train_temp[:, categorical_features]
    )
    x_train_temp[:, numerical_features] = mean_imputation(
        x_train_temp[:, numerical_features]
    )
    x_test_temp[:, categorical_features] = median_imputation(
        x_test_temp[:, categorical_features]
    )
    x_test_temp[:, numerical_features] = mean_imputation(
        x_test_temp[:, numerical_features]
    )

    if CONFIG["REMOVE_OUTLIERS"]:
        low_perc = np.percentile(x_train_temp, 2, axis=0)
        high_perc = np.percentile(x_train_temp, 100 - 2, axis=0)
        x_train_temp = remove_outliers(x_train_temp, low_perc, high_perc)
        x_test_temp = remove_outliers(x_test_temp, low_perc, high_perc)

    # Data engineering: ratios, log-transform, polynomial features
    if CONFIG["BUILD_RATIOS"]:
        print("Building ratios...")
        ratio_features_train = build_ratios(x_train_temp[:, numerical_features])
        ratio_features_test = build_ratios(x_test_temp[:, numerical_features])
        x_train_temp = np.hstack((x_train_temp, ratio_features_train))
        x_test_temp = np.hstack((x_test_temp, ratio_features_test))

    if CONFIG["BUILD_LOG"]:
        print("Building log...")
        log_features_train = build_log(x_train_temp[:, numerical_features])
        log_features_test = build_log(x_test_temp[:, numerical_features])
        x_train_temp = np.hstack((x_train_temp, log_features_train))
        x_test_temp = np.hstack((x_test_temp, log_features_test))

    if CONFIG["BUILD_POLY"]:
        print("Building polynomial features...")
        x_train_temp = build_poly(x_train_temp, 2)
        x_test_temp = build_poly(x_test_temp, 2)

    # Drop columns with a single unique value or highly correlated columns
    if CONFIG["DROP_SINGLE"]:
        x_train_temp, cols_to_keep_3 = drop_single_value_columns(x_train_temp)
        x_test_temp = x_test_temp[:, cols_to_keep_3]
        print("Dropping single valued columns...")

    x_train_temp, cols_to_keep_2 = drop_correlated_columns(
        x_train_temp, CONFIG["DROP_CORR_THRESHOLD"]
    )
    x_test_temp = x_test_temp[:, cols_to_keep_2]

    # Standardize the datasets
    if CONFIG["STANDARDIZE"]:
        print("Standardizing...")
        x_train_temp = standardize(x_train_temp)
        x_test_temp = standardize(x_test_temp)

    return x_train_temp, x_test_temp


def build_network(input_dim):
    network = [Fully_connected(input_dim, 128), ReLU(), Fully_connected(128, 2)]
    return network


def train_and_evaluate(network, x_train_temp, y_train):
    train_log = []

    for epoch in range(5):
        for x_batch, y_batch in iterate_minibatches(
            x_train_temp, y_train, batchsize=1000, shuffle=True
        ):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, x_train_temp) == y_train))

        f1_train = f1_score(y_train, predict(network, x_train_temp))

        print("Epoch", epoch)
        print("Train accuracy:", train_log[-1])
        print("F1 score train:", f1_train)

    return train_log


def main():
    X_train, x_test, y_train, train_ids, test_ids = load_and_split_data("../data")
    x_train_temp, x_test_temp = preprocess_data(X_train, x_test)
    network = build_network(x_train_temp.shape[1])
    train_and_evaluate(network, x_train_temp, y_train)

    y_pred = predict(network, x_test_temp)
    y_pred[np.where(y_pred == 0)] = -1
    create_csv_submission(test_ids, y_pred, "best_model.csv")


if __name__ == "__main__":
    main()
