# This is for INFSCI 2440 in Spring 2026.
# Task 1: Regression task 
import pandas as pd
from Features import Features as features
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 1================")
        return

    def model_1_run(self):
        print("Model 1: Linear Regression")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        mse = self.train_and_evaluate(LinearRegression())

        # Evaluate learned model on testing data, and print the results.
        print("*"*50)
        print("Mean squared error\t" + f"{mse:.4f}")
        return

    def model_2_run(self):
        print("--------------------\nModel 2: Support Vector Regression")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        # Define the hyperparameter grid to find the best model.
        param_grid = {
            'C': [2, 5, 7, 10],
            'epsilon': [0.01, 0.05, 0.1], 
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }

        mse = self.train_and_evaluate(SVR(), param_grid)

        # Evaluate learned model on testing data, and print the results.
        print("*"*50)
        print("Mean squared error\t" + f"{mse:.4f}")
        return

    ''' 
    Helper function to train and evaluate a model with optional hyperparameter tuning. 
    
    Returns:
        mse (float): The mean squared error on the test set.
    '''
    def train_and_evaluate(self, model, param_grid=None):
        # Apply preprocessing.
        df_train, df_test, _, _ = features().preprocess()

        # Apply One-Hot Encoding for nominal categories.  
        # (e.g. Mjob, Fjob, reason, guardian, school, sex, address, famsize, Pstatus)
        train_len = len(df_train)
        combined = pd.concat([df_train, df_test], axis=0)
        combined_encoded = pd.get_dummies(combined, drop_first=True) # Convert to binary columns.

        # Ensure consistency in train/test splits after encoding.
        df_train_final = combined_encoded.iloc[:train_len, :]
        df_test_final = combined_encoded.iloc[train_len:, :]

        # Drop final grade column (we want to predict this), and use everything else.
        x_train = df_train_final.drop('G3', axis=1)
        y_train = df_train_final['G3']
        x_test = df_test_final.drop('G3', axis=1)
        y_test = df_test_final['G3']

        # Scale all features (only improves SVR in this task).
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # If a grid is provided, find the best version of the model, otherwise fit the standard model.
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
            grid_search.fit(x_train_scaled, y_train)
            model = grid_search.best_estimator_
            print(f"Best Parameters: {grid_search.best_params_}")
        else:
            model = model
            model.fit(x_train_scaled, y_train)

        # Predict and calculate error.
        y_pred = model.predict(x_test_scaled)
        mse = mean_squared_error(y_test, y_pred)

        return mse