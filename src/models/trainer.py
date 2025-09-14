import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from ..utils.logger import get_logger, PerformanceTimer


class ModelTrainer:
    """
    Simple, direct model trainer that handles multiple algorithms.
    Follows the same approach as the original script.
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        
        # Store trained models and results
        self.trained_models = {}
        self.training_results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all enabled models from configuration.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with training results for each model
        """
        self.logger.info("Starting model training phase...")
        
        results = {}
        
        # Get enabled models from config
        for model_name, model_config in self.config.get("models", {}).items():
            if model_config.get("enabled", False):
                self.logger.info(f"Training {model_name}...")
                
                with PerformanceTimer(f"{model_name} training", self.logger):
                    model_result = self._train_single_model(
                        model_name, model_config, X_train, y_train
                    )
                    results[model_name] = model_result
        
        # Store results
        self.training_results = results
        
        # Find best model based on cross-validation score
        self._select_best_model()
        
        self.logger.info("Model training completed!")
        return results
    
    def _train_single_model(
        self, 
        model_name: str, 
        model_config: dict, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning."""
        
        # Create base model
        base_model = self._create_base_model(model_name, model_config["base_params"])
        
        # Get grid search parameters
        grid_params = model_config.get("grid_params", {})
        
        if grid_params:
            # Perform hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=grid_params,
                cv=self.config.get("cv_folds", 10),
                scoring=self.config.get("scoring_metric", "average_precision"),
                n_jobs=self.config.get("n_jobs", -1),
                verbose=self.config.get("verbose", 2)
            )
            
            # Fit with grid search
            grid_search.fit(X_train, y_train)
            
            # Store best model
            best_model = grid_search.best_estimator_
            self.trained_models[model_name] = best_model
            
            result = {
                "model": best_model,
                "best_params": grid_search.best_params_,
                "best_cv_score": grid_search.best_score_,
                "grid_search": grid_search
            }
            
            self.logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
            self.logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
            
        else:
            # Train without hyperparameter tuning
            base_model.fit(X_train, y_train)
            self.trained_models[model_name] = base_model
            
            result = {
                "model": base_model,
                "best_params": model_config["base_params"],
                "best_cv_score": None,
                "grid_search": None
            }
            
            self.logger.info(f"{model_name} - Trained with base parameters")
        
        return result
    
    def _create_base_model(self, model_name: str, base_params: dict):
        """Create base model instance."""
        if model_name == "logistic_regression":
            return LogisticRegression(**base_params)
        elif model_name == "random_forest":
            return RandomForestClassifier(**base_params)
        elif model_name == "svm":
            return SVC(**base_params)
        elif model_name == "xgboost":
            return XGBClassifier(**base_params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _select_best_model(self):
        """Select the best model based on cross-validation scores."""
        if not self.training_results:
            return
        
        best_score = -1
        best_name = None
        
        for model_name, result in self.training_results.items():
            cv_score = result.get("best_cv_score")
            if cv_score is not None and cv_score > best_score:
                best_score = cv_score
                best_name = model_name
        
        if best_name:
            self.best_model = self.trained_models[best_name]
            self.best_model_name = best_name
            self.logger.info(f"Best model: {best_name} (CV Score: {best_score:.4f})")
        else:
            # If no CV scores available, just pick the first model
            first_model = list(self.trained_models.keys())[0]
            self.best_model = self.trained_models[first_model]
            self.best_model_name = first_model
            self.logger.info(f"Selected model: {first_model} (no CV scores to compare)")
    
    def get_model(self, model_name: str):
        """Get a specific trained model."""
        return self.trained_models.get(model_name)
    
    def get_best_model(self):
        """Get the best performing model."""
        return self.best_model, self.best_model_name
    
    def save_models(self, save_dir: str):
        """Save all trained models to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_file = save_path / f"{model_name}_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved {model_name} model to {model_file}")
        
        # Save training results metadata
        results_file = save_path / "training_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump({
                "training_results": self.training_results,
                "best_model_name": self.best_model_name
            }, f)
        
        self.logger.info(f"All models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load trained models from disk."""
        load_path = Path(load_dir)
        
        # Load models
        for model_file in load_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            self.trained_models[model_name] = model
            self.logger.info(f"Loaded {model_name} model")
        
        # Load training results metadata
        results_file = load_path / "training_results.pkl"
        if results_file.exists():
            with open(results_file, "rb") as f:
                saved_data = pickle.load(f)
                self.training_results = saved_data["training_results"]
                self.best_model_name = saved_data["best_model_name"]
                if self.best_model_name in self.trained_models:
                    self.best_model = self.trained_models[self.best_model_name]
        
        self.logger.info(f"Models loaded from {load_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        summary = {
            "models_trained": list(self.trained_models.keys()),
            "best_model": self.best_model_name,
            "training_results": {}
        }
        
        for model_name, result in self.training_results.items():
            summary["training_results"][model_name] = {
                "best_cv_score": result.get("best_cv_score"),
                "best_params": result.get("best_params"),
                "has_grid_search": result.get("grid_search") is not None
            }
        
        return summary