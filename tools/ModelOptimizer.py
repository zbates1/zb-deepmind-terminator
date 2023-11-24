import os
import optuna
import numpy as np
import pandas as pd
import glob
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate

class ModelOptimizer:
    
    def __init__(self, label_data_path, dataset_path, save_path):
        """
        Initializes the ModelOptimizer class with paths to label data, dataset, and save location for results.

        :param label_data_path: Path to the CSV file containing the label data.
        :param dataset_path: Directory containing the feature data files (.npy files).
        :param save_path: Path where the optimization results will be saved.
        """
        self.label_data_path = label_data_path
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.results = {}
        self.model_mapping = {
            'RandomForest': RandomForestRegressor,
            'Ridge': Ridge,
            'XGBoost': XGBRegressor,
            'LinearRegression': LinearRegression
        }
        
    def smart_read_csv(self, filepath):
        """Read a CSV file with either comma or tab as delimiter."""
        # Try with comma first
        with open(filepath, 'r') as file:
            snippet = file.read(1024)  # read the first 1024 bytes to determine the delimiter
        # Determine the delimiter based on the snippet
        if '\t' in snippet and ',' not in snippet:
            delimiter = '\t'
        else:
            delimiter = ','  # default to comma if both are present or neither are present
        return pd.read_csv(filepath, delimiter=delimiter)

    def objective(self, trial, X, y):
        """
        Objective function for the Optuna study, used to find the best hyperparameters for various models.

        :param trial: An Optuna trial object.
        :param X: Feature data.
        :param y: Target data.
        :return: Mean test score after cross-validation.
        """
        classifier_name = trial.suggest_categorical('classifier', list(self.model_mapping.keys()))
        clf = None

        if classifier_name == 'XGBoost':
            xgb_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_jobs': -1
            }
            clf = XGBRegressor(**xgb_params)

        elif classifier_name == 'LinearRegression':
            clf = LinearRegression(n_jobs=-1)

        elif classifier_name == 'Ridge':
            ridge_params = {
                'alpha': trial.suggest_float('alpha', 0.1, 10.0)
            }
            clf = Ridge(**ridge_params)

        elif classifier_name == 'RandomForest':
            rf_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
            }
            clf = RandomForestRegressor(**rf_params, n_jobs=-1)
        
        return cross_validate(clf, X, y, cv=4, n_jobs=-1, return_train_score=True)['test_score'].mean()

    def optimize_hyperparameters(self, file_list, n_trials=10):
        label_df = self.smart_read_csv(self.label_data_path)
        y = label_df.iloc[:, 1].to_numpy()
        print('y shape: ', y.shape)
        
        for file in file_list:
            print(f'\nFile being processed in hyperparameter optimization:\n {file}')
            embeddings = np.load(file)
            X = embeddings  # Push to CPU (if necessary)
            
            if len(X) != len(y):
                print("X_train and y_train have different lengths, likely, this is due to the run_inference.py batching, so we will trim the y_train")
                y = y[:len(X)]
                
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)

            self.results[file] = {
                "Value": study.best_value,
                "Params": study.best_params
            }

        return self.results  # Changed to self.results

    def save_results_to_json(self, results, file_path):
        """
        Saves the optimization results to a JSON file.

        :param results: Dictionary containing the optimization results.
        :param file_path: Path where the results JSON file will be saved.
        """
        
        # Create dir if it doesn't exist
        file_path_dir = os.path.dirname(file_path)
        if not os.path.exists(file_path_dir):
            os.makedirs(file_path_dir)
            
        with open(file_path, 'w') as json_file:
            json.dump(results, json_file)
            print(f"Results saved to {file_path}")

    def optimization_wrapper(self):
        """
        Wrapper method to run the optimization process.
        """
        globbed_reduced_embeds_path = glob.glob(f'{self.dataset_path}*.npy')
        results = self.optimize_hyperparameters(globbed_reduced_embeds_path)
        self.save_results_to_json(results, self.save_path)