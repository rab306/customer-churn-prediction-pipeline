"""
Configuration settings for Customer Churn Prediction Pipeline
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data-related configuration"""
    target_column: str = "Exited"
    features_to_drop: List[str] = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    
    def __post_init__(self):
        if self.features_to_drop is None:
            # Complain dropped due to data leakage (near-perfect correlation with target)
            self.features_to_drop = ["RowNumber", "CustomerId", "Surname", "Complain"]


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    continuous_features: List[str] = None
    categorical_features: List[str] = None
    balance_thresholds: List[float] = None
    age_transformation: str = "boxcox"  # options: "none", "log", "boxcox"
    outlier_detection_threshold: float = 3.5
    outlier_features: List[str] = None
    
    def __post_init__(self):
        if self.continuous_features is None:
            self.continuous_features = ["CreditScore", "Age", "Balance", "EstimatedSalary", "Point Earned"]
        
        if self.categorical_features is None:
            self.categorical_features = [
                "Geography", "Gender", "NumOfProducts", "IsActiveMember", 
                "HasCrCard", "Tenure", "Satisfaction Score", "Card Type"
            ]
        
        if self.balance_thresholds is None:
            self.balance_thresholds = [0, 50000]  # Creates: Zero, 0-50K, 50K+
        
        if self.outlier_features is None:
            self.outlier_features = ["CreditScore", "Age", "Balance"]


@dataclass
class ModelConfig:
    """Model training configuration"""
    cv_folds: int = 10
    scoring_metric: str = "average_precision"
    n_jobs: int = -1
    verbose: int = 2
    
    # Model-specific configurations
    models: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "logistic_regression": {
                    "enabled": False,
                    "base_params": {"random_state": self.random_state},
                    "grid_params": {}
                },
                "random_forest": {
                    "enabled": True,
                    "base_params": {"random_state": self.random_state},
                    "grid_params": {
                        "n_estimators": [100, 200, 300],
                        "criterion": ["gini", "entropy"],
                        "max_depth": [10, 20, 30],
                        "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 6}]
                    }
                },
                "svm": {
                    "enabled": True,
                    "base_params": {"probability": True, "random_state": self.random_state},
                    "grid_params": {
                        "C": [0.1, 1, 10],
                        "kernel": ["poly", "rbf", "sigmoid"],
                        "gamma": ["scale", "auto"],
                        "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 6}]
                    }
                },
                "xgboost": {
                    "enabled": True,
                    "base_params": {
                        "random_state": self.random_state,
                        "use_label_encoder": False,
                        "eval_metric": "logloss",
                        "objective": "binary:logistic"
                    },
                    "grid_params": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 6, 8],
                        "learning_rate": [0.05, 0.1, 0.3],
                        "scale_pos_weight": [2, 4, 6]
                    }
                }
            }
    
    @property
    def random_state(self) -> int:
        return 42


@dataclass
class PathConfig:
    """Path configuration"""
    data_dir: str = "data"
    data_file: str = "Customer-Churn-Records.csv"  # Default filename
    output_dir: str = "results"
    model_dir: str = "models"
    plots_dir: str = "plots"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for attr_name in ["data_dir", "output_dir", "model_dir", "plots_dir", "logs_dir"]:
            path = getattr(self, attr_name)
            os.makedirs(path, exist_ok=True)
    
    @property
    def data_path(self) -> str:
        """Get full path to data file"""
        return os.path.join(self.data_dir, self.data_file)


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = None
    features: FeatureConfig = None
    models: ModelConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.paths is None:
            self.paths = PathConfig()


# Default configuration instance
def get_config() -> Config:
    """Factory function to get default configuration"""
    return Config()


# Environment-specific configurations
def get_config_for_env(env: str = "development") -> Config:
    """Get configuration for specific environment"""
    config = get_config()
    
    if env == "production":
        config.models.verbose = 0
        config.models.cv_folds = 5  # Faster for production
    elif env == "testing":
        config.models.cv_folds = 3
        # Smaller grid search for testing
        for model_name, model_config in config.models.models.items():
            if "grid_params" in model_config:
                # Take only first option for each param to speed up testing
                for param, values in model_config["grid_params"].items():
                    if isinstance(values, list) and len(values) > 1:
                        model_config["grid_params"][param] = [values[0]]
    
    return config