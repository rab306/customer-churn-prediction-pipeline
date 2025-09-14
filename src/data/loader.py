import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from ..utils.logger import get_logger
from .validator import DataValidator


class DataLoader:
    """
    Handles data loading and initial processing for churn prediction pipeline.
    
    Follows Single Responsibility Principle: Only responsible for loading and 
    basic data preparation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.validator = DataValidator(logger=self.logger)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            self.logger.info(f"Loading data from {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Log basic info
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Validate data
            self.validator.validate_dataframe(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Add target distribution if target column exists
        if "Exited" in df.columns:
            target_dist = df["Exited"].value_counts()
            info["target_distribution"] = {
                "counts": target_dist.to_dict(),
                "percentages": (target_dist / len(df) * 100).to_dict()
            }
        
        return info
    
    def log_data_summary(self, df: pd.DataFrame) -> None:
        """Log summary statistics of the dataset."""
        info = self.get_data_info(df)
        
        self.logger.info("=" * 50)
        self.logger.info("DATA SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Shape: {info['shape']}")
        self.logger.info(f"Missing values: {sum(info['missing_values'].values())}")
        self.logger.info(f"Duplicate rows: {info['duplicates']}")
        self.logger.info(f"Memory usage: {info['memory_usage'] / 1024**2:.2f} MB")
        
        if "target_distribution" in info:
            self.logger.info("Target distribution:")
            for label, count in info["target_distribution"]["counts"].items():
                pct = info["target_distribution"]["percentages"][label]
                self.logger.info(f"  {label}: {count} ({pct:.2f}%)")
    
    def clean_data(self, df: pd.DataFrame, features_to_drop: list) -> pd.DataFrame:
        """
        Perform initial data cleaning.
        
        Args:
            df: Input DataFrame
            features_to_drop: List of columns to drop
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Drop unnecessary features
        existing_drops = [col for col in features_to_drop if col in df_clean.columns]
        if existing_drops:
            df_clean = df_clean.drop(columns=existing_drops)
            self.logger.info(f"Dropped columns: {existing_drops}")
        
        # Log cleaning results
        self.logger.info(f"Data shape after cleaning: {df_clean.shape}")
        
        return df_clean
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Perform data quality checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            "completeness": {},
            "consistency": {},
            "validity": {}
        }
        
        # Completeness: Check missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        quality_report["completeness"] = {
            "columns_with_missing": missing_pct[missing_pct > 0].to_dict(),
            "total_missing_percentage": missing_pct.mean()
        }
        
        # Consistency: Check duplicates
        quality_report["consistency"] = {
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": df.duplicated().sum() / len(df) * 100
        }
        
        # Validity: Check data types and ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quality_report["validity"] = {
            "negative_values": {col: (df[col] < 0).sum() for col in numeric_cols},
            "infinite_values": {col: np.isinf(df[col]).sum() for col in numeric_cols}
        }
        
        return quality_report