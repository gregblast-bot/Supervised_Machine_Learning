# This is for INFSCI 2440 in Spring 2026.
# Task 3: Multi-label task 
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from Features import Features as features
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        return

    def model_1_run(self):
        print("Model 1: Logistic Regression")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        # Define the hyperparameter grid to find the best model.
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100],          
            'estimator__penalty': ['l1', 'l2'],          
            'estimator__solver': ['liblinear', 'saga']   
        }

        overall_accuracy, hamming_loss_value = self.train_and_evaluate(OneVsRestClassifier(LogisticRegression(max_iter=5000)), param_grid)

        # Evaluate learned model on testing data, and print the results.
        print("*"*50)
        print("Accuracy\t" + str(overall_accuracy) + "\tHamming loss\t" + str(hamming_loss_value))
        return

    def model_2_run(self):
        print("--------------------\nModel 2: Support Vector Machine")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        # Define the hyperparameter grid to find the best model.
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100],          
            'estimator__kernel': ['linear', 'rbf'],      
            'estimator__gamma': ['scale', 'auto', 0.1]   
        }

        overall_accuracy, hamming_loss_value = self.train_and_evaluate(OneVsRestClassifier(SVC()), param_grid)

        # Evaluate learned model on testing data, and print the results.
        print("*"*50)
        print("Accuracy\t" + str(overall_accuracy) + "\tHamming loss\t" + str(hamming_loss_value))
        return

    ''' 
    Helper function to train and evaluate a model with optional hyperparameter tuning. 
    
    Returns:
        overall_accuracy (float): The overall accuracy of the model on the test set.
        class_report (dict): A dictionary containing classification metrics for each class.
    '''
    def train_and_evaluate(self, model, param_grid=None):
        # Apply preprocessing.
        df_train, df_test, _, _ = features().preprocess()

        # Apply One-Hot Encoding for nominal categories except our target Mjob.  
        # (e.g. Mjob, Fjob, reason, guardian, school, sex, address, famsize, Pstatus)
        train_len = len(features().df_train)
        combined = pd.concat([df_train, df_test], axis=0)
        combined_encoded = pd.get_dummies(combined, drop_first=True) # Convert to binary columns.

        # Ensure consistency in train/test splits after encoding.
        df_train_final = combined_encoded.iloc[:train_len, :]
        df_test_final = combined_encoded.iloc[train_len:, :]

        # Drop mother's job column (we want to predict this), and use everything else.
        target_names = ['edu_school', 'edu_family', 'edu_paid', 'edu_no'] # The 4 binary labels we want to predict.
        x_data_train = df_train_final.drop(columns=target_names, axis=1)
        y_labels_train = df_train_final[target_names].values
        x_data_test = df_test_final.drop(columns=target_names, axis=1)
        y_labels_test = df_test_final[target_names].values

        # Scale all features (only improves SVR in this task).
        scaler = StandardScaler()
        x_data_train_scaled = scaler.fit_transform(x_data_train)
        x_data_test_scaled = scaler.transform(x_data_test)

        # If a grid is provided, find the best version of the model, otherwise fit the standard model.
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=0)
            grid_search.fit(x_data_train_scaled, y_labels_train)
            model = grid_search.best_estimator_
            print(f"Best Parameters: {grid_search.best_params_}")
        else:
            model = model
            model.fit(x_data_train_scaled, y_labels_train)

        # Predict and calculate metrics.
        y_labels_pred = model.predict(x_data_test_scaled)
        overall_accuracy = f'{accuracy_score(y_labels_test, y_labels_pred):.4f}'
        hamming_loss_value = f'{hamming_loss(y_labels_test, y_labels_pred):.4f}'

        # Count the labels to prove that this is multi-label-classification and not multi-category-classification.
        row_sums = np.sum(y_labels_train, axis=1)
        counts, values = np.unique(row_sums, return_counts=True)
        print("\nDistribution of tags per student:")
        for val, count in zip(counts, values):
            print(f"{int(val)} Tag(s): {count} students")
        multi_label_count = np.sum(row_sums > 1)
        print(f"Students with more than one support tag: {multi_label_count}")

        return overall_accuracy, hamming_loss_value