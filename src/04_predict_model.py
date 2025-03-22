import pandas as pd
import numpy as np
import argparse
import logging
import joblib
from pathlib import Path
from itertools import product
from sklearn.metrics import f1_score, roc_curve, recall_score
from utils import feature_subsets_snv, feature_subsets_indel, feature_subsets_all, extract_features
from sklearn.model_selection import KFold
from models.model import ML_model

def find_sensitivity_threshold(y_true, y_proba, target_sensitivity):
    """Find threshold that achieves target sensitivity on training data"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # Find the threshold closest to target sensitivity
    valid_idx = np.where(tpr >= target_sensitivity)[0]
    if len(valid_idx) == 0:
        return None  # Can't achieve target sensitivity
    optimal_idx = valid_idx[np.argmin(np.abs(tpr[valid_idx] - target_sensitivity))]
    return thresholds[optimal_idx]

def train_and_predict_holdout(X_train, y_train, X_holdout, y_holdout, model, info_holdout=None, 
                            repetition=1, target_sensitivity=None, model_save_path=None,
                            variant_type='indel'):
    """
    Train model using 5-fold CV for parameter selection, then predict on holdout set.
    Optionally adjust threshold to achieve target sensitivity.
    Optionally save the trained model to disk.
    
    Returns DataFrame with predictions and model evaluation metrics.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = [dict(zip(model.input_params.keys(), v)) 
                 for v in product(*model.input_params.values())]
    
    # Store average scores for each parameter set
    param_scores = []
    
    # Try each parameter combination
    for params in param_grid:
        fold_scores = []
        
        # Perform 5-fold CV for this parameter set
        for train_idx, val_idx in cv.split(X_train):
            # Get fold data - now using DataFrames
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            # Train and evaluate on this fold
            model.fit(X_fold_train, y_fold_train, **params)
            y_pred_val = model.predict(X_fold_val, feature_subset=params.get('feature_subset', 'all'))
            fold_scores.append(f1_score(y_fold_val, y_pred_val))
        
        # Store average score for this parameter set
        param_scores.append({
            'params': params,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        })
    
    # Find best parameters (using F1 score)
    best_result = max(param_scores, key=lambda x: x['mean_score'])
    best_params = best_result['params']
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"CV Score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")
    
    # Train final model on full training data with best parameters
    model.fit(X_train, y_train, **best_params)

    # Get predictions with F1-optimized threshold
    feature_subset = best_params.get('feature_subset', 'all')
    y_pred_orig = model.predict(X_holdout, feature_subset=feature_subset)
    y_proba = model.predict_proba(X_holdout, feature_subset=feature_subset)
    
    # Calculate original metrics
    orig_threshold = model.optimal_threshold
    orig_f1 = f1_score(y_holdout, y_pred_orig)
    orig_sensitivity = recall_score(y_holdout, y_pred_orig)
    
    # If target sensitivity is specified, set sensitivity threshold
    print(f"F1-optimal threshold: {model.optimal_threshold}")
    train_proba = model.predict_proba(X_train, feature_subset=feature_subset)
    sens_thresh = find_sensitivity_threshold(y_train, train_proba, target_sensitivity/100)
    print(f"Sensitivity threshold: {sens_thresh}")
    
    if target_sensitivity is not None:
        # Set the sensitivity threshold
        sensitivity_threshold = model.set_sensitivity_threshold(X_train, y_train, target_sensitivity/100)
        
        if sensitivity_threshold is not None:
            # Get predictions using sensitivity threshold
            y_pred_sens = model.predict(
                X_holdout, 
                feature_subset=feature_subset,
                use_sensitivity_threshold=True
            )
            new_f1 = f1_score(y_holdout, y_pred_sens)
            new_sensitivity = recall_score(y_holdout, y_pred_sens)
            
            logging.info(f"F1-optimal threshold: {orig_threshold:.4f}, Sensitivity threshold: {sensitivity_threshold:.4f}")
            logging.info(f"F1-optimal F1: {orig_f1:.4f}, Sensitivity-based F1: {new_f1:.4f}")
            logging.info(f"F1-optimal Sensitivity: {orig_sensitivity:.4f}, Sensitivity-based Sensitivity: {new_sensitivity:.4f}")
            
            # Use sensitivity-based predictions 
            y_pred = y_pred_sens
        else:
            logging.warning(f"Could not achieve target sensitivity {target_sensitivity}")
            y_pred = y_pred_orig
    else:
        # Use F1-optimal predictions
        y_pred = y_pred_orig

    if model_save_path:
        model_save_path = Path(model_save_path) / f"{model.model_type}_{variant_type}_model.joblib"
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        logging.info(f"Model feature names: {model.feature_names_}")
    
    # Collect predictions
    prediction_data = []
    for idx in range(len(y_holdout)):
        prediction_entry = {
            'repetition': repetition,
            'y_true': y_holdout[idx],
            'y_pred': y_pred[idx],
            'y_proba': y_proba[idx],
            'model_name': model.model_type,
            'best_parameter': best_params,
            'cv_score': best_result['mean_score'],
            'f1_threshold': orig_threshold,
            'target_sensitivity': target_sensitivity
        }
        
        # Add sensitivity threshold if available
        if target_sensitivity is not None and model.sensitivity_threshold is not None:
            prediction_entry['sensitivity_threshold'] = model.sensitivity_threshold
        
        if info_holdout is not None:
            prediction_entry.update(info_holdout.iloc[idx].to_dict())
        
        prediction_data.append(prediction_entry)
    
    return pd.DataFrame(prediction_data)

def main():
    parser = argparse.ArgumentParser(description='Run model training and prediction on hold-out data.')
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'xgboost', 'random_forest', 'gmm'],
                        help='Model type to use for training: logistic_regression, xgboost, random_forest, gmm')
    parser.add_argument('--all_models', action='store_true', help='Run the script for all models sequentially', default=True)
    parser.add_argument('--output', type=str, help='Path to save the output predictions file', default='/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/output/holdout_preds/E2LLC_2_F1_snv.tsv')
    parser.add_argument('--data', type=str, help='Path to the training dataset file', default='/Users/madsnielsen/Predisposed/data/datasets/E1LLC/processed_dataset_indel.tsv')
    parser.add_argument('--holdout', type=str, help='Path to the hold-out dataset file', default='/Users/madsnielsen/Predisposed/data/datasets/E2LHC/processed_dataset_indel.tsv')
    parser.add_argument('--vtype', type=str, help='Variant type to use for training: snv, indel, all', default='indel')
    parser.add_argument('--log', type=str, help='Path to the log file', default='train_model.log')
    parser.add_argument('--target_sensitivity', type=float, help='Sensitivity to optimise threshold for', default=99.99)
    parser.add_argument('--save_models', type=str, help='Directory to save trained models', default='/Users/madsnielsen/Predisposed/projects/filter_manuscript_repo/output/model_weights')

    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    target_sensitivity=args.target_sensitivity
    random_seed = 2
    dataset_path = Path(args.data)
    holdout_path = Path(args.holdout)
    output_path = Path(args.output)

    if args.save_models:
        model_save_path = Path(args.save_models)
        model_save_path.mkdir(exist_ok=True, parents=True)
        logging.info(f"Models will be saved to {model_save_path}")
    else:
        model_save_path = None

    if args.vtype == 'snv':
        feature_subsets = feature_subsets_snv
    elif args.vtype == 'indel':
        feature_subsets = feature_subsets_indel
    elif args.vtype == 'all':
        feature_subsets = feature_subsets_all
    
    X, y, _, feature_subsets_np = extract_features(dataset_path, feature_subsets)
    X_holdout, y_holdout, info_holdout, _ = extract_features(holdout_path, feature_subsets)

    model_configs = {
        'logistic_regression': {
            'params': {
                'feature_subset': ['all'],
                'C': [0.1, 1, 1.3, 1.5, 2, 10, 50],
                'max_iter': [10000],
                'penalty': ['l1', 'elasticnet', 'l2']
            },
            'init_params': {
                'class_weight': 'balanced',
                'transformation': 'standard'
            }
        },
        'xgboost': {
            'params': {
                'feature_subset': ['all'],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2, 0.3],
                'colsample_bytree': [0.65, 0.7, 0.8, 0.9]
            },
            'init_params': {
                'class_weight': 'balanced',
                'transformation': 'standard'
            }
        },
        'random_forest': {
            'params': {
                'feature_subset': ['all'],
                'n_estimators': [100],
                'max_depth': [12],
                'min_samples_leaf': [1],
            },
            'init_params': {
                'class_weight': 'balanced',
                'transformation': 'standard'
            }
        },
        'gmm': {
            'params': {
                'feature_subset': ['all'],
                'n_components': [3, 4, 5],
                'covariance_type': ['full']
            },
            'init_params': {
                'transformation': 'standard'
            }
        }
    }

    # Run for all models if --all_models is set
    if args.all_models:
        for model_name, config in model_configs.items():
            logging.info(f"Training {model_name}")
            print(f"Training {model_name}")

            model = ML_model(
                model_type=model_name,
                input_params=config['params'],
                available_subsets=feature_subsets_np,
                random_state=random_seed,
                **config['init_params']
            )

            prediction_df = train_and_predict_holdout(
                X, y, X_holdout, y_holdout, model, info_holdout, 
                target_sensitivity=target_sensitivity,
                model_save_path=model_save_path
            )
            
            output_path_model = output_path.with_name(f"{output_path.stem}_{model_name}{output_path.suffix}")
            prediction_df.to_csv(output_path_model, sep='\t', index=False)
    else:
        # Run for the specified model only
        if args.model:
            config = model_configs[args.model]
            model = ML_model(
                model_type=args.model,
                input_params=config['params'],
                available_subsets=feature_subsets_np,
                random_state=random_seed,
                **config['init_params']
            )
            prediction_df = train_and_predict_holdout(
                X, y, X_holdout, y_holdout, model, info_holdout,
                target_sensitivity=target_sensitivity,
                model_save_path=model_save_path
            )
            prediction_df.to_csv(output_path, sep='\t', index=False)
        else:
            print("Please specify a model with --model or use --all_models to run all models.")

if __name__ == '__main__':
    main()