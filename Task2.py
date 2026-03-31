# This is for INFSCI 2440 in Spring 2026.
# Task 2: Multi-category task 
import pandas as pd
from sklearn.calibration import LabelEncoder
from Features import Features as features
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class Task2:
    # add necessary comments to your code.
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        return

    def print_category_results(self, category, precision, recall, f1):
        print("*"*50)
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("*"*50)
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        print("Model 1: Decision Tree")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        # Define the hyperparameter grid to find the best model.
        param_grid = {
            'max_depth': [5, 10, 15, 20],     
            'min_samples_split': [2, 5, 10],          
            'min_samples_leaf': [1, 2, 4],           
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced'],
            'random_state': [42]
        }

        overall_accuracy, class_report = self.train_and_evaluate(DecisionTreeClassifier(), param_grid)
        macro_metrics = class_report['macro avg']
        
        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(overall_accuracy, f"{macro_metrics['precision']:.4f}", f"{macro_metrics['recall']:.4f}", f"{macro_metrics['f1-score']:.4f}")
        categories = ["teacher", "health", "services", "at_home", "other"]
        for category in categories:
            metrics = class_report.get(category) # Extract metrics for the specific category.
            if metrics:
                self.print_category_results(category, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1-score']:.4f}")
            else:
                print(f"Category {category} not found in classification report.")
        return

    def model_2_run(self):
        print("--------------------\nModel 2: Nearest Neighbors")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        # Define the hyperparameter grid to find the best model.
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11], 
            'weights': ['uniform', 'distance'],   
            'metric': ['euclidean', 'manhattan'],  
            'leaf_size': [20, 30, 40]           
        }

        overall_accuracy, class_report = self.train_and_evaluate(KNeighborsClassifier(), param_grid)
        macro_metrics = class_report['macro avg']

        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(overall_accuracy, f"{macro_metrics['precision']:.4f}", f"{macro_metrics['recall']:.4f}", f"{macro_metrics['f1-score']:.4f}")
        categories = ["teacher", "health", "services", "at_home", "other"]
        for category in categories:
            metrics = class_report.get(category) # Extract metrics for the specific category.
            if metrics:
                self.print_category_results(category, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1-score']:.4f}"   )
            else:
                print(f"Category {category} not found in classification report.")
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
        # (e.g. Fjob, reason, guardian, school, sex, address, famsize, Pstatus)
        train_len = len(df_train)
        combined = pd.concat([df_train, df_test], axis=0)
        features_to_encode = ['Fjob', 'reason', 'guardian']
        combined_encoded = pd.get_dummies(combined, columns=features_to_encode, drop_first=True) # Convert to binary columns.

        # Ensure consistency in train/test splits after encoding.
        df_train_final = combined_encoded.iloc[:train_len, :]
        df_test_final = combined_encoded.iloc[train_len:, :]

        # Drop mother's job column (we want to predict this), and use everything else.
        x_data_train = df_train_final.drop('Mjob', axis=1)
        y_labels_train = df_train_final['Mjob']
        x_data_test = df_test_final.drop('Mjob', axis=1)
        y_labels_test = df_test_final['Mjob']

        # KNN requires label encoding for the target variable.
        target_names = None
        if isinstance(model, KNeighborsClassifier):
            le = LabelEncoder() # Initialize the encoder
            y_labels_train = le.fit_transform(y_labels_train)
            y_labels_test = le.transform(y_labels_test)
            target_names = le.classes_

        # Scale all features (only improves SVR in this task).
        scaler = StandardScaler()
        x_data_train_scaled = scaler.fit_transform(x_data_train)
        x_data_test_scaled = scaler.transform(x_data_test)

        # If a grid is provided, find the best version of the model, otherwise fit the standard model.
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', verbose=0) # cv=10 for 10-fold-cross-validation.
            grid_search.fit(x_data_train_scaled, y_labels_train)
            model = grid_search.best_estimator_
            print(f"Best Parameters: {grid_search.best_params_}")
        else:
            model = model
            model.fit(x_data_train_scaled, y_labels_train)

        # Predict and calculate metrics.
        y_labels_pred = model.predict(x_data_test_scaled)
        overall_accuracy = accuracy_score(y_labels_test, y_labels_pred)
        class_report = classification_report(y_labels_test, y_labels_pred, target_names=target_names, output_dict=True, zero_division=0)

        return overall_accuracy, class_report