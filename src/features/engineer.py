"""
Feature engineering for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
import scipy.stats as ss
from typing import Tuple, List, Optional, Dict, Any
import logging

from ..utils.logger import get_logger, PerformanceTimer


class OutlierDetector:
    """
    Detects outliers using Modified Z-Score method.
    
    Uses Median Absolute Deviation for robustness to skewed distributions.
    """
    
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
    
    def modified_z_score(self, series: pd.Series) -> pd.Series:
        """
        Calculate modified Z-score using Median Absolute Deviation.
        
        Args:
            series: Input series
            
        Returns:
            Modified Z-scores
        """
        median = series.median()
        mad = ss.median_abs_deviation(series, scale='normal')
        
        # Avoid division by zero
        if mad == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        return 0.6745 * (series - median) / mad
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            
        Returns:
            Dictionary with outlier information for each column
        """
        outlier_info = {}
        
        for col in columns:
            if col in df.columns:
                z_scores = self.modified_z_score(df[col])
                outlier_mask = abs(z_scores) > self.threshold
                
                outlier_info[col] = {
                    "count": outlier_mask.sum(),
                    "percentage": (outlier_mask.sum() / len(df)) * 100,
                    "indices": df[outlier_mask].index.tolist(),
                    "values": df.loc[outlier_mask, col].tolist()
                }
        
        return outlier_info


class FeatureTransformer:
    """
    Handles feature transformations for normalization and distribution improvement.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.transformation_params = {}
    
    def apply_age_transformation(
        self, 
        df: pd.DataFrame, 
        method: str = "boxcox"
    ) -> pd.DataFrame:
        """
        Apply transformation to Age feature to improve normality.
        
        Args:
            df: Input DataFrame
            method: Transformation method ("none", "log", "boxcox")
            
        Returns:
            DataFrame with transformed Age feature
        """
        if "Age" not in df.columns:
            self.logger.warning("Age column not found, skipping transformation")
            return df
        
        df_transformed = df.copy()
        
        if method == "log":
            df_transformed["Age_LogTransformed"] = np.log(df["Age"])
            self.logger.info("Applied log transformation to Age")
            
        elif method == "boxcox":
            # Box-Cox transformation
            transformed_values, lambda_param = ss.boxcox(df["Age"])
            df_transformed["Age_BoxCoxTransformed"] = transformed_values
            
            # Store parameter for inverse transformation if needed
            self.transformation_params["age_boxcox_lambda"] = lambda_param
            
            self.logger.info(f"Applied Box-Cox transformation to Age (λ={lambda_param:.4f})")
            
        elif method == "none":
            self.logger.info("No transformation applied to Age")
        else:
            raise ValueError(f"Unknown transformation method: {method}")
        
        return df_transformed
    
    def create_balance_categories(
        self, 
        df: pd.DataFrame, 
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Categorize Balance feature based on thresholds.
        
        Args:
            df: Input DataFrame
            thresholds: List of threshold values [low_threshold, high_threshold]
            
        Returns:
            DataFrame with Balance_Category feature
        """
        if thresholds is None:
            thresholds = [0, 50000]
        
        if "Balance" not in df.columns:
            self.logger.warning("Balance column not found, skipping categorization")
            return df
        
        df_categorized = df.copy()
        
        def categorize_balance(balance):
            if balance <= thresholds[0]:
                return 'Zero'
            elif balance <= thresholds[1]:
                return '0-50K'
            else:
                return '50K+'
        
        df_categorized['Balance_Category'] = df['Balance'].apply(categorize_balance)
        
        # Log category distribution
        category_counts = df_categorized['Balance_Category'].value_counts()
        category_percentages = (category_counts / len(df_categorized)) * 100
        
        self.logger.info("Balance categorization completed:")
        for category, count in category_counts.items():
            pct = category_percentages[category]
            self.logger.info(f"  {category}: {count} ({pct:.1f}%)")
        
        return df_categorized
    
    def calculate_skewness(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
        """
        Calculate skewness for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate skewness for
            
        Returns:
            Dictionary with skewness values
        """
        skewness_values = {}
        
        for col in columns:
            if col in df.columns:
                skewness_values[col] = df[col].skew()
        
        return skewness_values


class FeatureEngineer:
    """
    Main feature engineering class that orchestrates all transformations.
    
    Implements the Open-Closed Principle: open for extension (new transformations)
    but closed for modification of existing functionality.
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        
        # Initialize components
        self.outlier_detector = OutlierDetector(
            threshold=config.get("outlier_detection_threshold", 3.5)
        )
        self.transformer = FeatureTransformer(logger=self.logger)
        
        # Track feature engineering steps
        self.engineering_steps = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        with PerformanceTimer("Feature Engineering", self.logger):
            df_engineered = df.copy()
            
            # Step 1: Outlier detection (for analysis purposes)
            self._detect_and_log_outliers(df_engineered)
            
            # Step 2: Age transformation
            df_engineered = self._apply_age_transformation(df_engineered)
            
            # Step 3: Balance categorization
            df_engineered = self._create_balance_categories(df_engineered)
            
            # Step 4: Feature selection (remove original/intermediate features)
            df_engineered = self._select_final_features(df_engineered)
            
            # Step 5: Log engineering summary
            self._log_engineering_summary(df, df_engineered)
            
            return df_engineered
    
    def _detect_and_log_outliers(self, df: pd.DataFrame) -> None:
        """Detect and log outliers for analysis."""
        outlier_features = self.config.get("outlier_features", [])
        
        if outlier_features:
            self.logger.info("Detecting outliers...")
            outlier_info = self.outlier_detector.detect_outliers(df, outlier_features)
            
            for feature, info in outlier_info.items():
                self.logger.info(
                    f"{feature}: {info['count']} outliers ({info['percentage']:.2f}%)"
                )
            
            self.engineering_steps.append("outlier_detection")
    
    def _apply_age_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply age transformation based on configuration."""
        transformation_method = self.config.get("age_transformation", "boxcox")
        
        if transformation_method != "none":
            self.logger.info(f"Applying {transformation_method} transformation to Age")
            df = self.transformer.apply_age_transformation(df, transformation_method)
            self.engineering_steps.append(f"age_transformation_{transformation_method}")
        
        return df
    
    def _create_balance_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance categories based on configuration."""
        balance_thresholds = self.config.get("balance_thresholds", [0, 50000])
        
        self.logger.info("Creating balance categories")
        df = self.transformer.create_balance_categories(df, balance_thresholds)
        self.engineering_steps.append("balance_categorization")
        
        return df
    
    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final features for modeling."""
        df_final = df.copy()
        
        # Remove intermediate age transformations, keeping only the best one
        age_transformation = self.config.get("age_transformation", "boxcox")
        
        columns_to_drop = []
        
        if age_transformation == "boxcox":
            # Keep BoxCox, drop original and log
            columns_to_drop.extend(["Age", "Age_LogTransformed"])
        elif age_transformation == "log":
            # Keep log, drop original and BoxCox
            columns_to_drop.extend(["Age", "Age_BoxCoxTransformed"])
        elif age_transformation == "none":
            # Keep original, drop transformations
            columns_to_drop.extend(["Age_LogTransformed", "Age_BoxCoxTransformed"])
        
        # Remove original Balance (we have Balance_Category)
        columns_to_drop.append("Balance")
        
        # Drop columns that exist
        existing_drops = [col for col in columns_to_drop if col in df_final.columns]
        if existing_drops:
            df_final = df_final.drop(columns=existing_drops)
            self.logger.info(f"Dropped intermediate features: {existing_drops}")
        
        self.engineering_steps.append("feature_selection")
        
        return df_final
    
    def _log_engineering_summary(self, df_original: pd.DataFrame, df_final: pd.DataFrame) -> None:
        """Log summary of feature engineering process."""
        self.logger.info("=" * 50)
        self.logger.info("FEATURE ENGINEERING SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Original shape: {df_original.shape}")
        self.logger.info(f"Final shape: {df_final.shape}")
        self.logger.info(f"Steps applied: {', '.join(self.engineering_steps)}")
        
        # Log skewness improvements for Age (if transformed)
        age_transformation = self.config.get("age_transformation", "boxcox")
        if age_transformation != "none":
            original_skew = df_original["Age"].skew() if "Age" in df_original.columns else None
            
            if age_transformation == "boxcox" and "Age_BoxCoxTransformed" in df_final.columns:
                final_skew = df_final["Age_BoxCoxTransformed"].skew()
            elif age_transformation == "log" and "Age_LogTransformed" in df_final.columns:
                final_skew = df_final["Age_LogTransformed"].skew()
            else:
                final_skew = None
            
            if original_skew is not None and final_skew is not None:
                self.logger.info(f"Age skewness: {original_skew:.4f} → {final_skew:.4f}")
        
        self.logger.info("Feature engineering completed successfully")
    
    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with feature information
        """
        continuous_features = [col for col in self.config.get("continuous_features", []) 
                             if col in df.columns]
        categorical_features = [col for col in self.config.get("categorical_features", []) 
                              if col in df.columns]
        
        # Add engineered features
        if "Age_BoxCoxTransformed" in df.columns:
            continuous_features.append("Age_BoxCoxTransformed")
        if "Age_LogTransformed" in df.columns:
            continuous_features.append("Age_LogTransformed")
        if "Balance_Category" in df.columns:
            categorical_features.append("Balance_Category")
        
        return {
            "total_features": len(df.columns),
            "continuous_features": continuous_features,
            "categorical_features": categorical_features,
            "engineering_steps": self.engineering_steps,
            "transformation_params": self.transformer.transformation_params
        }