import pandas as pd
import numpy as np
import argparse
import logging
import yaml
from pathlib import Path
from itertools import product
from sklearn.metrics import f1_score
from utils import feature_subsets_snv, feature_subsets_indel, feature_subsets_all, extract_features
from sklearn.model_selection import KFold
from models.model import ML_model

## mads - 2025-03-23
# Script for retraining model on narrow set of hyperparameters
# and perform the last and final prediction on held-out dataset.

def load_model_configs(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / "model_configs.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_and_predict_holdout(X_train, y_train, X_holdout, y_holdout, model, info_holdout=None, 
                            repetition=1, target_sensitivity=None, model_save_path=None):
    """Train model using CV, select best parameters, and predict holdout set."""
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = [dict(zip(model.input_params.keys(), v)) for v in product(*model.input_params.values())]
    
    param_scores = []

    for params in param_grid:
        fold_scores = [
            f1_score(y_train[val_idx], model.fit(X_train.iloc[train_idx], y_train[train_idx], **params)
                     .predict(X_train.iloc[val_idx], feature_subset=params.get('feature_subset', 'all')))
            for train_idx, val_idx in cv.split(X_train)
        ]
        
        param_scores.append({
            'params': params,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        })
    
    best_params = max(param_scores, key=lambda x: x['mean_score'])['params']
    logging.info(f"Best parameters: {best_params}")

    # Final model training with best params
    model.fit(X_train, y_train, **best_params)

    # Predictions and probabilities
    y_pred_orig = model.predict(X_holdout, feature_subset=best_params.get('feature_subset', 'all'))
    y_proba = model.predict_proba(X_holdout, feature_subset=best_params.get('feature_subset', 'all'))
    
    orig_threshold = model.optimal_threshold

    logging.info(f"F1-optimal threshold: {orig_threshold}")

    if target_sensitivity is not None:
        # Set sensitivity threshold
        sensitivity_threshold = model.set_sensitivity_threshold(X_train, y_train, target_sensitivity / 100)
        
        if sensitivity_threshold is not None:
            y_pred_sens = model.predict(X_holdout, feature_subset=best_params.get('feature_subset', 'all'), use_sensitivity_threshold=True)
            logging.info(f"Sensitivity threshold: {sensitivity_threshold}")
        else:
            logging.warning(f"Could not achieve target sensitivity {target_sensitivity}")
            y_pred_sens = y_pred_orig
    else:
        y_pred_sens = y_pred_orig
    
    if model_save_path:
        model_save_path = Path(model_save_path) / f"{model.model_type}_model.joblib"
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")

    # Construct prediction DataFrame
    prediction_data = [
        {
            'repetition': repetition,
            'y_true': y_holdout[idx],
            'y_pred': y_pred_sens[idx],
            'y_proba': y_proba[idx],
            'model_name': model.model_type,
            'best_parameter': best_params,
            'cv_score': max(param_scores, key=lambda x: x['mean_score'])['mean_score'],
            'f1_threshold': orig_threshold,
            'target_sensitivity': target_sensitivity
        }
        for idx in range(len(y_holdout))
    ]

    if info_holdout is not None:
        for idx, entry in enumerate(prediction_data):
            entry.update(info_holdout.iloc[idx].to_dict())

    return pd.DataFrame(prediction_data)

def main():
    parser = argparse.ArgumentParser(description='Run model training and prediction on hold-out data.')
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'xgboost', 'random_forest', 'gmm'],
                        help='Model type to use for training: logistic_regression, xgboost, random_forest, gmm',
                        default=None)
    parser.add_argument('--output', type=str, help='Path to save the output predictions file', required=True)
    parser.add_argument('--data', type=str, help='Path to the training dataset file', required=True)
    parser.add_argument('--holdout', type=str, help='Path to the hold-out dataset file', required=True)
    parser.add_argument('--vtype', type=str, help='Variant type to use for training: snv, indel, all', default='snv')
    parser.add_argument('--log', type=str, help='Path to the log file', default='train_model.log')
    parser.add_argument('--target_sensitivity', type=float, help='Sensitivity to optimise threshold for', default=99.99)
    parser.add_argument('--save_models', type=str, help='Directory to save trained models', default='./model_weights')

    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO)

    model_configs = load_model_configs()

    dataset_path, holdout_path, output_path = Path(args.data), Path(args.holdout), Path(args.output)
    
    feature_subsets = {'snv': feature_subsets_snv, 'indel': feature_subsets_indel, 'all': feature_subsets_all}[args.vtype]
    X, y, _, feature_subsets_np = extract_features(dataset_path, feature_subsets)
    X_holdout, y_holdout, info_holdout, _ = extract_features(holdout_path, feature_subsets)

    model_names = model_configs.keys() if not args.model else [args.model]

    for model_name in model_names:
        config = model_configs[model_name]
        model = ML_model(
            model_type=model_name,
            input_params=config['params'],
            available_subsets=feature_subsets_np,
            random_state=2,
            **config['init_params']
        )

        prediction_df = train_and_predict_holdout(
            X, y, X_holdout, y_holdout, model, info_holdout,
            target_sensitivity=args.target_sensitivity,
            model_save_path=args.save_models
        )

        output_path_model = output_path.with_name(f"{output_path.stem}_{model_name}{output_path.suffix}")
        prediction_df.to_csv(output_path_model, sep='\t', index=False)

if __name__ == '__main__':
    main()