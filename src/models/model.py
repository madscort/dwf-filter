import numpy as np
from copy import deepcopy
import joblib
import os
from pathlib import Path
from itertools import product
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, precision_recall_curve

class ML_model:
    def __init__(
        self,
        model_type='logistic_regression',
        input_params=None,
        transformation='standard',
        random_state=42,
        available_subsets=None,
        class_weight='balanced'
    ):
        self.model_type = model_type
        self.input_params = input_params
        self.random_state = random_state
        self.transformation = transformation
        self.available_subsets = available_subsets
        self.class_weight = class_weight
        self.optimal_threshold = 0.5
        self.scaler = None
        
    def _preprocess(self, X, subset=None, fit=False):
        """Preprocess data with feature selection and scaling"""
        # Feature selection
        if subset is not None and subset != 'all':
            if subset not in self.available_subsets:
                raise ValueError(f"Unknown feature subset: {subset}")
            X = X[:, self.available_subsets[subset]]
            
        # Scaling
        if self.transformation:
            if fit:
                if self.transformation == 'standard':
                    self.scaler = StandardScaler()
                elif self.transformation == 'yeo-johnson':
                    self.scaler = PowerTransformer(method='yeo-johnson')
                else:
                    raise ValueError("Transformation must be 'standard' or 'yeo-johnson'")
                return self.scaler.fit_transform(X)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call fit() first.")
                return self.scaler.transform(X)
        return X
    
    def _compute_optimal_threshold(self, y_true, y_proba):
        """Compute optimal classification threshold using PR curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Add 0 to thresholds for edge case where all predictions are negative
        thresholds = np.append(thresholds, 1)
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        # Return threshold that maximizes F1 score
        return thresholds[np.argmax(f1_scores)]

    def fit(self, X, y, **params):
        """Fit model with preprocessing and class weights"""
        # Preprocess features
        X_processed = self._preprocess(X, params.get('feature_subset'), fit=True)
        
        # Initialize model with appropriate parameters
        if self.model_type == 'logistic_regression':
            if params.get('penalty') == 'elasticnet':
                l1_ratio = 0.5
            else:
                l1_ratio = None
            
            self.model = LogisticRegression(
                C=params.get('C', 1),
                max_iter=params.get('max_iter', 1000),
                solver=params.get('solver', 'saga'),
                penalty=params.get('penalty', 'l2'),
                random_state=self.random_state,
                l1_ratio=l1_ratio,
                class_weight=self.class_weight
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=self.random_state,
                class_weight=self.class_weight
            )
            
        elif self.model_type == 'xgboost':
            if self.class_weight == 'balanced':
                scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
            else:
                scale_pos_weight = 1.0
                
            self.model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.1),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state
            )
            
        elif self.model_type == 'lightgbm':
            if self.class_weight == 'balanced':
                class_weights = 'balanced'
            else:
                class_weights = None
                
            self.model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.1),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                class_weight=class_weights,
                random_state=self.random_state
            )
            
        elif self.model_type == 'gmm':
            # For GMM, we train on positive class only and use density estimation
            X_positive = X_processed[y == 1]
            self.model = GaussianMixture(
                n_components=params.get('n_components', 5),
                covariance_type=params.get('covariance_type', 'full'),
                random_state=self.random_state
            )
            self.model.fit(X_positive)
            
            # Compute optimal threshold using all training data
            scores = self.model.score_samples(X_processed)
            self.optimal_threshold = self._compute_optimal_threshold(y, scores)
            return self
            
        # Fit model
        self.model.fit(X_processed, y)
        
        # Compute optimal threshold for classification (except GMM)
        if self.model_type != 'gmm':
            y_proba = self.predict_proba(X, params.get('feature_subset'))
            self.optimal_threshold = self._compute_optimal_threshold(y, y_proba)
            
        return self

    def predict(self, X, feature_subset=None, threshold=None):
        """Make predictions with optional custom threshold"""
        if threshold is None:
            threshold = self.optimal_threshold
            
        X_processed = self._preprocess(X, feature_subset)
        
        if self.model_type == 'gmm':
            scores = self.model.score_samples(X_processed)
            return (scores >= threshold).astype(int)
        else:
            proba = self.predict_proba(X, feature_subset)
            return (proba >= threshold).astype(int)

    def predict_proba(self, X, feature_subset=None):
        """Predict probabilities or scores"""
        X_processed = self._preprocess(X, feature_subset)
        
        if self.model_type == 'gmm':
            return self.model.score_samples(X_processed)
        else:
            return self.model.predict_proba(X_processed)[:, 1]
            
    def search_hyperparameters(self, X, y):
        """Search for best hyperparameters"""
        best_score = -np.inf
        best_params = None
        best_model = None

        param_grid = [dict(zip(self.input_params.keys(), values)) 
                     for values in product(*self.input_params.values())]

        for params in param_grid:
            self.fit(X, y, **params)
            
            # Use optimal threshold for evaluation
            y_pred = self.predict(X, feature_subset=params.get('feature_subset'))
            score = f1_score(y, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                if self.model_type == 'gmm':
                    best_model = (deepcopy(self.model), self.optimal_threshold)
                else:
                    best_model = deepcopy(self.model)

        return best_model, best_params, best_score

    def save(self, filepath):
        """Save the model to disk"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk"""
        return joblib.load(filepath)
    
    def save_metadata(self, filepath):
        """Save model metadata to a text file"""
        with open(filepath, 'w') as f:
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Optimal Threshold: {self.optimal_threshold}\n")
            f.write(f"Transformation: {self.transformation}\n")
            
            if self.best_params:
                f.write("\nBest Parameters:\n")
                for key, value in self.best_params.items():
                    f.write(f"- {key}: {value}\n")
                
            if hasattr(self.model, 'feature_importances_'):
                f.write("\nFeature Importances:\n")
                if self.best_params and self.best_params.get('feature_subset', None) != 'all':
                    subset = self.best_params['feature_subset']
                    feature_indices = self.available_subsets[subset]
                    for i, importance in enumerate(self.model.feature_importances_):
                        f.write(f"- Feature {feature_indices[i]}: {importance:.6f}\n")
                else:
                    for i, importance in enumerate(self.model.feature_importances_):
                        f.write(f"- Feature {i}: {importance:.6f}\n")