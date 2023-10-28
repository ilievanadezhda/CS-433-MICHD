# ML Project 1

## Introduction

This repository contains the code and resources for our Machine Learning project. The project focuses on building a model that can predict the risk of getting MICHD.

## Documentation

### Data

- The data was sourced from https://www.cdc.gov/brfss/annual_data/annual_2015.html.

### Feature Preprocessing

- Pre-processing steps included cleaning, imputation, outlier removal and standardization. 
- The exact implementations are located in `feature_processing.py` file contains the methods used for feature extraction and processing.

### Feature Expansion

- We used three methods for feature expansion, `build_log`, `build_poly` and `build_ratios`. These three methods are used to expand upon the existing features. They can be found in `feature_preprocessing.py` as well.

### Cross-Validation

- Cross-validation was implemented to validate the robustness of our model.
- The `cross_validation.py` file contains the methods and logic for cross-validation.
- Additionally, for the neural network, there is a `cross_validation_nn` function that can be found in `neural_network.py` file.
  
### Neural Network

- For our best results, we implemented a simple neural network from scratch.
- Details and implementation can be found in `neural_network.py`.

### Implementations and Additional Code

- As required, we implemented six distinct machine learning models: linear regression using gradient de- scent, linear regression using stochastic gradient de- scent, least squares regression using normal equations, ridge regression using normal equations, logistic re- gression using gradient descent and regularized logistic regression using gradient descent, these can be found in `implementations.py` and `implementations_utils.py`.
- The `helpers.py` was provided from the staff. It contains some helper functions, such as loading csv data, or creating submissions.

## Notebooks and Additional Code

- `EDA.ipynb`: Exploratory Data Analysis of the dataset. Figure out which features are continuous, which ones are categorical. Check if there are obvious relationships between the features.
- `reg_logistic_regression.py`: This file contains the random search for the best hyperparameters for regularized logistic regression.
- `logistic_regression.py`: This file contains the random search for the best hyperparameters for logistic regression
- `experiments.ipynb`: Contains various experiments conducted during the project. It was mainly used to pinpoint the best hyperparameters.
- `neural_network.ipynb`: Notebook for the neural network model. This model contains the workflow of the neural network. The difference is that we use a separate validation set which we used in order to find optimal number of epochs.
- `pipeline.ipynb`: This file contains a pipeline for data preprocessing and model evaluation on logistic regression and regularized logistic regression.

## Running the Code
To execute the main pipeline, follow the steps below:

1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:

As an output you should get a best_model.csv file in the same directory.

```bash
python run.py
```
## Dependencies and Versioning

- **Python**: 3.9.16
- **NumPy**: 1.23.5

> Note: Using NumPy version 1.19 yields inconsistent results when executing `run.py`. We are not aware whether earlier versions might produce the same incosistencies.
