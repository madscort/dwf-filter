import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, precision_recall_curve, recall_score, roc_curve
import joblib
import os

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
        self.optimal_threshold = 0.5  # F1-optimized threshold (default)
        self.sensitivity_threshold = None  # Sensitivity-optimized threshold
        self.target_sensitivity = None  # Target sensitivity value
        self.scaler = None
        self.best_params = None
        
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
        """Compute optimal classification threshold using PR curve (maximizes F1)"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Add 0 to thresholds for edge case where all predictions are negative
        thresholds = np.append(thresholds, 1)
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        # Return threshold that maximizes F1 score
        return thresholds[np.argmax(f1_scores)]
    
    def _compute_sensitivity_threshold(self, y_true, y_proba, target_sensitivity):
        """Compute threshold that achieves target sensitivity"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        # Find the threshold closest to target sensitivity
        valid_idx = np.where(tpr >= target_sensitivity)[0]
        if len(valid_idx) == 0:
            return None  # Can't achieve target sensitivity
        optimal_idx = valid_idx[np.argmin(np.abs(tpr[valid_idx] - target_sensitivity))]
        return thresholds[optimal_idx]

    def set_sensitivity_threshold(self, X, y, target_sensitivity):
        """Set threshold to achieve target sensitivity on provided data"""
        self.target_sensitivity = target_sensitivity
        y_proba = self.predict_proba(X, self.best_params.get('feature_subset', None))
        self.sensitivity_threshold = self._compute_sensitivity_threshold(y, y_proba, target_sensitivity)
        return self.sensitivity_threshold

    def fit(self, X, y, **params):
        """Fit model with preprocessing and class weights"""
        # Store the parameters for later use
        self.best_params = params
        
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
                min_samples_leaf=params.get('min_samples_leaf', 1),
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

    def predict(self, X, feature_subset=None, threshold=None, use_sensitivity_threshold=False):
        """
        Make predictions with specified threshold options
        
        Args:
            X: Features to predict on
            feature_subset: Subset of features to use
            threshold: Custom threshold to use (overrides other options)
            use_sensitivity_threshold: If True, use sensitivity threshold, otherwise use F1 threshold
            
        Returns:
            Binary predictions
        """
        if threshold is not None:
            final_threshold = threshold
        elif use_sensitivity_threshold and self.sensitivity_threshold is not None:
            final_threshold = self.sensitivity_threshold
        else:
            # Default to F1-optimized threshold
            final_threshold = self.optimal_threshold
            
        X_processed = self._preprocess(X, feature_subset)
        
        if self.model_type == 'gmm':
            scores = self.model.score_samples(X_processed)
            return (scores >= final_threshold).astype(int)
        else:
            proba = self.predict_proba(X, feature_subset)
            return (proba >= final_threshold).astype(int)

    def predict_proba(self, X, feature_subset=None):
        """Predict probabilities or scores"""
        X_processed = self._preprocess(X, feature_subset)
        
        if self.model_type == 'gmm':
            return self.model.score_samples(X_processed)
        else:
            return self.model.predict_proba(X_processed)[:, 1]
    
    def get_available_thresholds(self):
        """Get information about available thresholds"""
        thresholds = {
            'f1_optimal': self.optimal_threshold
        }
        
        if self.sensitivity_threshold is not None:
            thresholds['sensitivity'] = {
                'threshold': self.sensitivity_threshold,
                'target': self.target_sensitivity
            }
            
        return thresholds
            
    def save(self, filepath):
        """Save the model to disk"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk """
        return joblib.load(filepath)
