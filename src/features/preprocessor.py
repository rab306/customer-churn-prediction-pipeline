"""
Data preprocessing for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging
import pickle
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger, PerformanceTimer


class EncodingStrategy:
    """
    Handles encoding of categorical variables.
    
    Uses different strategies based on cardinality:
    - Binary features: Label encoding
    - Multi-class features: One-hot encoding
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.label_encoders = {}
        self.encoded_columns = []
        self.encoding_info = {}
    
    def fit_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Fit encoders and transform categorical columns.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in df.columns:
                self.logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            unique_values = df[col].nunique()
            self.encoding_info[col] = {
                "unique_values": unique_values,
                "original_values": df[col].unique().tolist()
            }
            
            if unique_values == 2:
                # Binary encoding: use LabelEncoder
                self._apply_label_encoding(df_encoded, col)
            else:
                # Multi-class encoding: use One-hot encoding
                df_encoded = self._apply_onehot_encoding(df_encoded, col)
        
        return df_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df_encoded = df.copy()
        
        # Apply label encodings
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories by mapping to most frequent category
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col])
                except ValueError:
                    self.logger.warning(f"Unseen categories in {col}, using fallback encoding")
                    df_encoded[col] = self._handle_unseen_categories(df_encoded[col], encoder)
        
        # For one-hot encoded columns, ensure all columns exist
        expected_columns = [col for col in self.encoded_columns if col not in self.label_encoders.keys()]
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Default value for missing one-hot columns
        
        return df_encoded
    
    def _apply_label_encoding(self, df: pd.DataFrame, column: str) -> None:
        """Apply label encoding for binary categorical variables."""
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        self.label_encoders[column] = encoder
        self.encoded_columns.append(column)
        
        self.logger.info(f"Applied label encoding to '{column}' (2 categories)")
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply one-hot encoding for multi-class categorical variables."""
        df_encoded = pd.get_dummies(df, columns=[column], prefix=column)
        
        # Track new columns
        new_columns = [col for col in df_encoded.columns if col.startswith(f"{column}_")]
        self.encoded_columns.extend(new_columns)
        
        self.logger.info(f"Applied one-hot encoding to '{column}' ({len(new_columns)} categories)")
        
        return df_encoded
    
    def _handle_unseen_categories(self, series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        """Handle unseen categories by mapping to most frequent category."""
        # Get the most frequent category from training
        most_frequent_encoded = 0  # Assuming 0 is always present in binary encoding
        
        # Create a copy and replace unseen values
        series_copy = series.copy()
        mask = ~series_copy.isin(encoder.classes_)
        
        if mask.any():
            self.logger.warning(f"Found {mask.sum()} unseen categories, mapping to most frequent")
            # Map unseen categories to the first class
            series_copy[mask] = encoder.classes_[0]
            series_copy = encoder.transform(series_copy)
        else:
            series_copy = encoder.transform(series_copy)
        
        return series_copy
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Get information about applied encodings."""
        return {
            "encoded_columns": self.encoded_columns,
            "label_encoded": list(self.label_encoders.keys()),
            "encoding_details": self.encoding_info
        }


class DataSplitter:
    """
    Handles train-test splitting with proper stratification.
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Get split parameters from config
        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", 42)
        stratify = y if self.config.get("stratify", True) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Log split information
        self._log_split_info(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    def _log_split_info(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> None:
        """Log information about data split."""
        self.logger.info("=" * 40)
        self.logger.info("DATA SPLIT SUMMARY")
        self.logger.info("=" * 40)
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        
        # Log class distribution
        train_dist = y_train.value_counts(normalize=True) * 100
        test_dist = y_test.value_counts(normalize=True) * 100
        
        self.logger.info("Class distribution:")
        for class_label in sorted(y_train.unique()):
            train_pct = train_dist.get(class_label, 0)
            test_pct = test_dist.get(class_label, 0)
            self.logger.info(f"  Class {class_label}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")


class DataScaler:
    """
    Handles feature scaling using StandardScaler.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.scaling_stats = {}
    
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler on training data and transform.
        
        Args:
            X_train: Training features DataFrame
            
        Returns:
            Scaled training features as numpy array
        """
        self.feature_names = X_train.columns.tolist()
        
        # Fit and transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Store scaling statistics
        self.scaling_stats = {
            "feature_means": self.scaler.mean_.tolist(),
            "feature_stds": self.scaler.scale_.tolist(),
            "feature_names": self.feature_names
        }
        
        self.logger.info(f"Fitted scaler on {len(self.feature_names)} features")
        
        return X_train_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted scaler.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Scaled features as numpy array
        """
        if self.scaler.mean_ is None:
            raise ValueError("Scaler has not been fitted yet")
        
        # Ensure same feature order
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.scaler.transform(X)
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """Get information about scaling parameters."""
        return self.scaling_stats


class DataPreprocessor:
    """
    Main preprocessing class that orchestrates all preprocessing steps.
    
    Implements the Facade pattern to provide a simple interface to complex subsystems.
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        
        # Initialize components
        self.encoder = EncodingStrategy(logger=self.logger)
        self.splitter = DataSplitter(config, logger=self.logger)
        self.scaler = DataScaler(logger=self.logger)
        
        # Track preprocessing state
        self.is_fitted = False
        self.preprocessing_info = {}
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline: encode, split, and scale.
        
        Args:
            df: Input DataFrame with engineered features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        with PerformanceTimer("Data Preprocessing", self.logger):
            
            # Step 1: Encode categorical variables
            df_encoded = self._encode_features(df)
            
            # Step 2: Split data
            X_train, X_test, y_train, y_test = self._split_data(df_encoded)
            
            # Step 3: Scale features
            X_train_scaled = self._scale_features(X_train, fit=True)
            X_test_scaled = self._scale_features(X_test, fit=False)
            
            # Update state
            self.is_fitted = True
            self._store_preprocessing_info(df, df_encoded, X_train)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Scaled features as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet")
        
        # Apply same preprocessing steps
        df_encoded = self.encoder.transform(df.drop(columns=[self.config["target_column"]], errors='ignore'))
        X_scaled = self.scaler.transform(df_encoded)
        
        return X_scaled
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        # Get categorical columns from config
        categorical_columns = [col for col in self.config.get("categorical_features", []) 
                             if col in df.columns]
        
        # Add Balance_Category if it exists (created during feature engineering)
        if "Balance_Category" in df.columns:
            categorical_columns.append("Balance_Category")
        
        # Also check for any remaining object columns that weren't in the config
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_columns:
            if col not in categorical_columns and col != self.config.get("target_column"):
                categorical_columns.append(col)
                self.logger.info(f"Found additional categorical column: {col}")
        
        if categorical_columns:
            self.logger.info(f"Encoding categorical features: {categorical_columns}")
            df_encoded = self.encoder.fit_transform(df, categorical_columns)
        else:
            df_encoded = df.copy()
            self.logger.info("No categorical features to encode")
        
        return df_encoded
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        target_column = self.config["target_column"]
        return self.splitter.split_data(df, target_column)
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale features."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def _store_preprocessing_info(
        self, 
        df_original: pd.DataFrame, 
        df_encoded: pd.DataFrame, 
        X_train: pd.DataFrame
    ) -> None:
        """Store information about preprocessing steps."""
        self.preprocessing_info = {
            "original_shape": df_original.shape,
            "encoded_shape": df_encoded.shape,
            "final_features": X_train.columns.tolist(),
            "encoding_info": self.encoder.get_encoding_info(),
            "scaling_info": self.scaler.get_scaling_info()
        }
    
    def save_preprocessors(self, save_dir: str) -> None:
        """Save fitted preprocessors to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save encoders
        if self.encoder.label_encoders:
            with open(save_path / "label_encoders.pkl", "wb") as f:
                pickle.dump(self.encoder.label_encoders, f)
        
        # Save scaler
        with open(save_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler.scaler, f)
        
        # Save preprocessing info
        with open(save_path / "preprocessing_info.pkl", "wb") as f:
            pickle.dump(self.preprocessing_info, f)
        
        self.logger.info(f"Preprocessors saved to {save_dir}")
    
    def load_preprocessors(self, load_dir: str) -> None:
        """Load fitted preprocessors from disk."""
        load_path = Path(load_dir)
        
        # Load encoders if they exist
        encoder_path = load_path / "label_encoders.pkl"
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                self.encoder.label_encoders = pickle.load(f)
        
        # Load scaler
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler.scaler = pickle.load(f)
        
        # Load preprocessing info
        info_path = load_path / "preprocessing_info.pkl"
        if info_path.exists():
            with open(info_path, "rb") as f:
                self.preprocessing_info = pickle.load(f)
        
        self.is_fitted = True
        self.logger.info(f"Preprocessors loaded from {load_dir}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing pipeline."""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "preprocessing_info": self.preprocessing_info,
            "feature_count": len(self.preprocessing_info.get("final_features", [])),
            "encoding_applied": bool(self.encoder.encoded_columns),
            "scaling_applied": self.scaler.scaler.mean_ is not None
        }