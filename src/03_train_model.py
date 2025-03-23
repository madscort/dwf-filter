import pandas as pd
import numpy as np
from copy import deepcopy
import sys
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from models.model import ML_model
from utils import feature_subsets_snv, feature_subsets_indel, feature_subsets_all, extract_features
import argparse
import logging

def inner_cross_validation(X_train, y_train, model, inner_folds=5, random_seed=42):
    """
    Perform inner cross-validation for model selection.
    """
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    
    best_model = None
    best_parameter = None
    best_score = -np.inf
    best_threshold = None

    # Inner loop for model selection
    for inner_train_index, inner_val_index in inner_cv.split(X_train):
        X_inner_train, y_inner_train = X_train.iloc[inner_train_index], y_train[inner_train_index]
        X_inner_val, y_inner_val = X_train.iloc[inner_val_index], y_train[inner_val_index]
        
        # Search for best hyperparameters
        current_model, current_parameter, current_score = model.search_hyperparameters(
            X_inner_train, y_inner_train
        )
        
        # Test the model on the inner validation set using the optimal threshold
        if model.model_type == 'gmm':
            current_model, optimal_threshold = current_model
            model.model = current_model
            model.optimal_threshold = optimal_threshold
        else:
            model.model = current_model
            model.optimal_threshold = model.optimal_threshold
            
        score = f1_score(
            y_inner_val, 
            model.predict(X_inner_val, feature_subset=current_parameter.get('feature_subset', None))
        )
        
        # Update best model if current model is better
        if score > best_score:
            if model.model_type == 'gmm':
                best_model = (deepcopy(current_model), optimal_threshold)
            else:
                best_model = deepcopy(current_model)
            best_parameter = current_parameter
            best_score = score
            best_threshold = model.optimal_threshold
    
    return best_model, best_parameter, best_score

def outer_cross_validation(X, y, model, outer_folds=5, inner_folds=5, random_state=42, info=None, repetition=1):
    """
    Perform outer cross-validation to evaluate the best model.
    """
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    outer_results = []
    prediction_data = []

    # Iterate over each outer fold
    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X)):
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        info_test = None if info is None else info.iloc[test_index]
        
        # Use inner cross-validation for model selection
        best_model, best_parameter, best_inner_score = inner_cross_validation(
            X_train, y_train, model, inner_folds, random_state
        )
        
        # Set the best model and threshold
        if model.model_type == 'gmm':
            model.model, model.optimal_threshold = best_model
        else:
            model.model = best_model
                
        # Evaluate the best model on the outer test set
        y_pred = model.predict(X_test, feature_subset=best_parameter.get('feature_subset', None))
        y_proba = model.predict_proba(X_test, feature_subset=best_parameter.get('feature_subset', None))

        # Collect prediction data
        for idx in range(len(y_test)):
            prediction_entry = {
                'fold': fold_idx + 1,
                'repetition': repetition,
                'y_true': y_test[idx],
                'y_pred': y_pred[idx],
                'y_proba': y_proba[idx],
                'model_type': model.model_type,
                'best_parameter': best_parameter,
                'optimal_threshold': model.optimal_threshold
            }
            if info_test is not None:
                prediction_entry.update(info_test.iloc[idx].to_dict())
            prediction_data.append(prediction_entry)

        # Store outer fold results
        outer_test_score = f1_score(y_test, y_pred)
        outer_results.append({
            'fold': fold_idx + 1,
            'outer_test_score': outer_test_score,
            'best_inner_score': best_inner_score,
            'best_parameter': best_parameter,
            'optimal_threshold': model.optimal_threshold
        })
    
    prediction_df = pd.DataFrame(prediction_data)
    return outer_results, prediction_df


def main(model="gmm", vtype="snv", log_path="output/test.log", dataset_path=None, output_path="output/testpreds.tsv"):
    """
    Main function to handle model training and evaluation.

    Arguments:
    - model: str, model type (default: "gmm")
    - vtype: str, variant type ('snv', 'indel', 'all')
    - log_path: str, path to the log file
    - dataset_path: str, path to the dataset file
    - output_path: str, path to save the output predictions file
    """

    # Define fixed dataset paths for each variant type
    default_dataset_paths = {
        "snv": "/Users/madsnielsen/Predisposed/data/datasets/E1LLC/processed_dataset_snv.tsv",
        "indel": "/Users/madsnielsen/Predisposed/data/datasets/E1LLC/processed_dataset_indel.tsv",
        "all": "/Users/madsnielsen/Predisposed/data/datasets/E1LLC/processed_dataset.tsv",
    }

    default_dataset_paths = {
        "snv": "/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/input/dataset_snv.tsv",
        "indel": "/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/input/dataset_indel.tsv",
        "all": "/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/input/dataset.tsv",
    }

    # If no dataset path is provided, use the default for the given variant type
    if dataset_path is None:
        if vtype in default_dataset_paths:
            dataset_path = default_dataset_paths[vtype]
        else:
            raise ValueError(f"Invalid variant type: {vtype}")

    # Set up logging
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # Select feature subsets based on `vtype`
    if vtype == 'snv':
        feature_subsets = feature_subsets_snv
    elif vtype == 'indel':
        feature_subsets = feature_subsets_indel
    elif vtype == 'all':
        feature_subsets = feature_subsets_all
    else:
        raise ValueError(f"Invalid variant type: {vtype}")
    
    X, y, info, feature_subsets_np = extract_features(dataset_path, feature_subsets)    
    outer_folds=10
    inner_folds=5
    repetitions=5
    random_seed=2

    LN_params = {
        'class_weight': ['balanced'],
        'feature_subset': ['all'],
        'C': [0.1, 1, 1.3, 1.5, 2],
        'max_iter': [10000],
        'penalty': ['l1', 'elasticnet', 'l2']
    }
    LN_model = ML_model('logistic_regression', LN_params, available_subsets=feature_subsets_np, random_state=random_seed)

    xgb_params = {
        'class_weight': ['balanced'],
        'feature_subset': ['all'],
        'n_estimators': [200, 250],
        'max_depth': [8,9],
        'learning_rate': [0.2, 0.3],
        'colsample_bytree': [0.8]
    }
    XGB_model = ML_model('xgboost', xgb_params, random_state=random_seed, available_subsets=feature_subsets_np)
    
    lgbm_params = {
        'feature_subset': ['all'],
        'class_weight': ['balanced'],
        'n_estimators': [12, 25, 50, 100],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.2, 0.5, 0.9],
        'colsample_bytree': [0.8],
    }

    lgbm_params = {}


    LGBM_model = ML_model('lightgbm', lgbm_params, random_state=random_seed, available_subsets=feature_subsets_np)

    gmm_params = {
        'feature_subset': ['normal_loose'],
        'class_weight': ['balanced'],
        'n_components': [1,2,4,6,8,10,12],
        'covariance_type': ['full'],
    }
    GM_model = ML_model('gmm', gmm_params, transformation='standard', random_state=random_seed, available_subsets=feature_subsets_np)


    rf_params = {
        'feature_subset': ['all'],
        'class_weight': ['balanced'],
        'n_estimators': [150,200],
        'max_depth': [12, 14, 16],
        'min_samples_leaf': [8],
    }
    RF_model = ML_model('random_forest', rf_params, random_state=random_seed, available_subsets=feature_subsets_np)

    model_mapping = {
            'logistic_regression': LN_model,
            'xgboost': XGB_model,
            'lightgbm': LGBM_model,
            'random_forest': RF_model,
            'gmm': GM_model
    }

    selected_model = model_mapping[model]
    models = [selected_model]
    all_predictions = []
    for model in models:
        logging.info(f"Running model training and evaluation for model: {model.model_type}")
        print(f"Running model training and evaluation for model: {model.model_type}")
        for i in range(repetitions):
            print(f"Running repetition {i + 1}/{repetitions}...")
            logging.info(f"Running repetition {i + 1}/{repetitions}...")
            
            random_state_repetition = random_seed + i 
            results, prediction_df = outer_cross_validation(X, y, model, outer_folds, inner_folds, random_state_repetition, info, repetition=i+1)
            
            print(f"Results for repetition {i + 1}:")
            logging.info(f"Results for repetition {i + 1}:")
            for j, result in enumerate(results):
                print(f"Fold {j + 1}: Inner Score: {result['best_inner_score']:.4f}, Outer Score: {result['outer_test_score']:.4f}, Best Parameter: {result['best_parameter']}")
                logging.info(f"Fold {j + 1}: Inner Score: {result['best_inner_score']:.4f}, Outer Score: {result['outer_test_score']:.4f}, Best Parameter: {result['best_parameter']}")
            all_predictions.append(prediction_df)
    
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'xgboost','lightgbm', 'random_forest', 'gmm'],
                        help='Model type to use for training: logistic_regression, xgboost, lightgbm, random_forest, gmm', default='gmm')
    parser.add_argument('--vtype', type=str, help='Variant type to use for training: snv, indel, all', default='snv')
    parser.add_argument('--log', type=str, help='Path to the log file', default='output/test.log')
    parser.add_argument('--data', type=str, help='Path to the dataset file', default=None)
    parser.add_argument('--output', type=str, help='Path to save the output predictions file', default='output/testpreds.tsv')

    args = parser.parse_args()

    main(
        model='gmm',
        vtype='indel',
        log_path='test.log',
        output_path='/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/output/test_preds/training_predictions_gmm_indel.tsv'
    )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run model training and evaluation.')
#     parser.add_argument('--model', type=str, choices=['logistic_regression', 'xgboost', 'random_forest', 'gmm'],
#                         help='Model type to use for training: logistic_regression, xgboost, random_forest, gmm', default='gmm')
#     parser.add_argument('--vtype', type=str, help='Variant type to use for training: snv, indel, all', default='snv')
#     parser.add_argument('--log', type=str, help='Path to the log file', default='output/test.log')
#     parser.add_argument('--data', type=str, help='Path to the dataset file', default=None)
#     parser.add_argument('--output', type=str, help='Path to save the output predictions file', default='output/testpreds.tsv')

#     args = parser.parse_args()

#     main(
#         model=args.model,
#         vtype=args.vtype,
#         log_path=args.log,
#         dataset_path=args.data,
#         output_path=args.output
#     )