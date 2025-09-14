"""
Data validation for Customer Churn Prediction pipeline
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ..utils.logger import get_logger


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class DataValidator:
    """
    Validates data quality and structure for churn prediction pipeline.
    
    Implements validation rules specific to customer churn datasets.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        
        # Expected columns for churn dataset
        self.expected_columns = {
            "required": ["Exited"],  # Target column must exist
            "optional": [
                "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
                "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Exited", "Complain",
                "Satisfaction Score", "Card Type", "Point Earned"
            ]
        }
        
        # Expected data types
        self.expected_dtypes = {
            "CreditScore": [np.number],
            "Age": [np.number],
            "Tenure": [np.number],
            "Balance": [np.number],
            "NumOfProducts": [np.number],
            "EstimatedSalary": [np.number],
            "Exited": [np.number, 'int64', 'int32'],
            "Geography": ['object'],
            "Gender": ['object']
        }
        
        # Valid ranges for numerical columns
        self.valid_ranges = {
            "CreditScore": (300, 850),    # Typical credit score range
            "Age": (10, 120),             # Reasonable age range
            "Tenure": (0, 50),            # Years with bank
            "Balance": (0, float('inf')), # Account balance
            "NumOfProducts": (1, 10),     # Number of products
            "Exited": (0, 1)              # Binary target
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Comprehensive validation of the input DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info("Starting data validation...")
            
            # Basic structure validation
            self._validate_structure(df)
            
            # Column validation
            self._validate_columns(df)
            
            # Data type validation
            self._validate_dtypes(df)
            
            # Range validation
            self._validate_ranges(df)
            
            # Business logic validation
            self._validate_business_rules(df)
            
            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            error_msg = f"Data validation failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg)
    
    def _validate_structure(self, df: pd.DataFrame) -> None:
        """Validate basic DataFrame structure"""
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        if df.shape[0] < 100:
            self.logger.warning(f"Dataset is very small: {df.shape[0]} rows")
        
        if df.shape[1] < 5:
            raise DataValidationError(f"Too few columns: {df.shape[1]}")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist"""
        missing_required = []
        for col in self.expected_columns["required"]:
            if col not in df.columns:
                missing_required.append(col)
        
        if missing_required:
            raise DataValidationError(f"Missing required columns: {missing_required}")
        
        # Log available columns
        available_optional = [col for col in self.expected_columns["optional"] 
                            if col in df.columns]
        self.logger.info(f"Available optional columns: {len(available_optional)}")
    
    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """Validate data types of columns"""
        for col, expected_types in self.expected_dtypes.items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                
                # Check if actual dtype matches any expected type
                type_match = any(
                    np.issubdtype(actual_dtype, expected_type) if hasattr(expected_type, '__name__')
                    else str(actual_dtype) == expected_type
                    for expected_type in expected_types
                )
                
                if not type_match:
                    self.logger.warning(
                        f"Column '{col}' has unexpected dtype: {actual_dtype}. "
                        f"Expected one of: {expected_types}"
                    )
    
    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate numerical column ranges"""
        for col, (min_val, max_val) in self.valid_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                
                # Check for values outside valid range
                outside_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                
                if outside_range > 0:
                    pct_outside = (outside_range / len(df)) * 100
                    if pct_outside > 5:  # More than 5% outside range
                        raise DataValidationError(
                            f"Column '{col}': {outside_range} values ({pct_outside:.1f}%) "
                            f"outside valid range [{min_val}, {max_val}]"
                        )
                    else:
                        self.logger.warning(
                            f"Column '{col}': {outside_range} values outside expected range"
                        )
    
    def _validate_business_rules(self, df: pd.DataFrame) -> None:
        """Validate business logic rules specific to churn prediction"""
        
        # Target variable validation
        if "Exited" in df.columns:
            unique_values = df["Exited"].unique()
            if not set(unique_values).issubset({0, 1}):
                raise DataValidationError(
                    f"Target variable 'Exited' contains invalid values: {unique_values}"
                )
            
            # Check class imbalance
            class_distribution = df["Exited"].value_counts(normalize=True)
            minority_class_pct = class_distribution.min() * 100
            
            if minority_class_pct < 5:
                self.logger.warning(
                    f"Severe class imbalance detected: minority class {minority_class_pct:.1f}%"
                )
            elif minority_class_pct < 10:
                self.logger.info(
                    f"Class imbalance detected: minority class {minority_class_pct:.1f}%"
                )
        
        # Geographic distribution
        if "Geography" in df.columns:
            geo_counts = df["Geography"].value_counts()
            if len(geo_counts) < 2:
                self.logger.warning("Only one geographic region present in data")
        
        # Product count validation
        if "NumOfProducts" in df.columns:
            max_products = df["NumOfProducts"].max()
            if max_products > 5:
                self.logger.warning(f"Unusually high number of products: {max_products}")
        
        # Age-tenure consistency
        if "Age" in df.columns and "Tenure" in df.columns:
            impossible_tenure = (df["Tenure"] > df["Age"] - 16).sum()
            if impossible_tenure > 0:
                self.logger.warning(
                    f"{impossible_tenure} records have tenure > (age - 16)"
                )
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing validation results
        """
        report = {
            "timestamp": pd.Timestamp.now(),
            "dataset_shape": df.shape,
            "validation_status": "passed",
            "issues": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Run all validations and collect issues
            self._validate_structure(df)
            self._validate_columns(df)
            self._validate_dtypes(df)
            self._validate_ranges(df)
            self._validate_business_rules(df)
            
        except DataValidationError as e:
            report["validation_status"] = "failed"
            report["issues"].append(str(e))
        
        # Add summary statistics
        report["summary"] = {
            "missing_values_total": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns)
        }
        
        return report