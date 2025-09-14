import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve
)

from ..utils.logger import get_logger


class ModelEvaluator:
    """
    Simple evaluator that matches the metrics from the original script.
    Focuses on precision-recall metrics for imbalanced churn data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.evaluation_results = {}
    
    def evaluate_model(
        self, 
        model, 
        model_name: str,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Precision-recall metrics (important for imbalanced data)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba, average='weighted')
        
        # ROC metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        except:
            roc_auc = None
            fpr, tpr, roc_thresholds = None, None, None
        
        result = {
            "model_name": model_name,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "average_precision": avg_precision,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "predictions": y_pred,
            "prediction_probabilities": y_pred_proba
        }
        
        self.evaluation_results[model_name] = result
        
        # Log key metrics
        self.logger.info(f"{model_name} Results:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Average Precision: {avg_precision:.4f}")
        if roc_auc:
            self.logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return result
    
    def evaluate_multiple_models(
        self, 
        trained_models: Dict[str, Any],
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models and compare results.
        
        Args:
            trained_models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results for each model
        """
        self.logger.info("Starting model evaluation phase...")
        
        results = {}
        
        for model_name, model in trained_models.items():
            result = self.evaluate_model(model, model_name, X_test, y_test)
            results[model_name] = result
        
        # Log comparison
        self._log_model_comparison(results)
        
        return results
    
    def _log_model_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Log comparison of all models."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("=" * 60)
        
        # Create comparison table data
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": f"{result['accuracy']:.4f}",
                "Avg Precision": f"{result['average_precision']:.4f}",
                "ROC AUC": f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
            })
        
        # Log table
        for data in comparison_data:
            self.logger.info(f"{data['Model']:20} | Acc: {data['Accuracy']} | AP: {data['Avg Precision']} | ROC: {data['ROC AUC']}")
    
    def get_best_model_by_metric(
        self, 
        metric: str = "average_precision"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, evaluation_result)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_score = -1
        best_model = None
        best_result = None
        
        for model_name, result in self.evaluation_results.items():
            score = result.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model = model_name
                best_result = result
        
        self.logger.info(f"Best model by {metric}: {best_model} ({best_score:.4f})")
        return best_model, best_result
    
    def print_detailed_report(self, model_name: str):
        """Print detailed evaluation report for a specific model."""
        if model_name not in self.evaluation_results:
            self.logger.error(f"No evaluation results for {model_name}")
            return
        
        result = self.evaluation_results[model_name]
        
        print(f"\n{'=' * 50}")
        print(f"{model_name.upper()} DETAILED EVALUATION")
        print(f"{'=' * 50}")
        
        # Basic metrics
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Average Precision: {result['average_precision']:.4f}")
        if result['roc_auc']:
            print(f"ROC AUC: {result['roc_auc']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(result['confusion_matrix'])
        
        # Classification report
        print(f"\nClassification Report:")
        class_report = result['classification_report']
        for class_label in ['0', '1']:  # Non-churn, Churn
            if class_label in class_report:
                metrics = class_report[class_label]
                print(f"  Class {class_label}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results."""
        if not self.evaluation_results:
            return {"status": "no_evaluations"}
        
        summary = {
            "models_evaluated": list(self.evaluation_results.keys()),
            "metrics_summary": {}
        }
        
        # Extract key metrics for each model
        for model_name, result in self.evaluation_results.items():
            summary["metrics_summary"][model_name] = {
                "accuracy": result["accuracy"],
                "average_precision": result["average_precision"],
                "roc_auc": result["roc_auc"]
            }
        
        # Find best models by different metrics
        best_by_accuracy = max(self.evaluation_results.items(), 
                              key=lambda x: x[1]["accuracy"])
        best_by_precision = max(self.evaluation_results.items(), 
                               key=lambda x: x[1]["average_precision"])
        
        summary["best_models"] = {
            "by_accuracy": best_by_accuracy[0],
            "by_average_precision": best_by_precision[0]
        }
        
        return summary
    
    def save_evaluation_results(self, save_path: str):
        """Save evaluation results to file."""
        import pickle
        
        with open(save_path, "wb") as f:
            pickle.dump(self.evaluation_results, f)
        
        self.logger.info(f"Evaluation results saved to {save_path}")
    
    def load_evaluation_results(self, load_path: str):
        """Load evaluation results from file."""
        import pickle
        
        with open(load_path, "rb") as f:
            self.evaluation_results = pickle.load(f)
        
        self.logger.info(f"Evaluation results loaded from {load_path}")