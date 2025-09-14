"""
Customer Churn Prediction Pipeline - Main Entry Point
"""
import argparse
from pathlib import Path
import sys

from config.settings import get_config
from src.utils.logger import setup_logging
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.features.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Path to the churn dataset CSV file"
    )
    parser.add_argument(
        "--config_env", 
        type=str, 
        default="development",
        choices=["development", "testing", "production"],
        help="Configuration environment"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate pre-trained models (skip training)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pre-trained models for evaluation only"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO", f"churn_pipeline_{args.config_env}")
    logger.info("Starting Customer Churn Prediction Pipeline")
    
    # Load configuration
    config = get_config()
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.paths.data_path
        logger.info(f"Using default data path: {data_path}")
    
    # Check if data file exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please either:")
        logger.info(f"1. Place your CSV file at: {data_path}")
        logger.info(f"2. Use --data_path argument to specify custom location")
        sys.exit(1)
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        loader = DataLoader()
        df = loader.load_data(data_path)
        loader.log_data_summary(df)
        
        # Clean data (remove unnecessary columns)
        df_clean = loader.clean_data(df, config.data.features_to_drop)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        engineer = FeatureEngineer(config.features.__dict__)
        df_engineered = engineer.engineer_features(df_clean)
        
        # Step 3: Preprocessing
        logger.info("Step 3: Data preprocessing...")
        preprocessor = DataPreprocessor(config.data.__dict__)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df_engineered)
        
        if args.evaluate_only:
            # Evaluation only mode
            if not args.model_path:
                logger.error("--model_path is required when using --evaluate_only")
                sys.exit(1)
            
            if not Path(args.model_path).exists():
                logger.error(f"Model path not found: {args.model_path}")
                sys.exit(1)
            
            logger.info("Evaluation-only mode: Loading pre-trained models...")
            trainer = ModelTrainer(config.models.__dict__)
            trainer.load_models(args.model_path)
            
            if not trainer.trained_models:
                logger.error("No models found in the specified path")
                sys.exit(1)
                
            logger.info(f"Loaded models: {list(trainer.trained_models.keys())}")
            
        else:
            # Full training mode
            # Step 4: Model training
            logger.info("Step 4: Model training...")
            trainer = ModelTrainer(config.models.__dict__)
            training_results = trainer.train_all_models(X_train, y_train)
            
            # Save trained models
            trainer.save_models(config.paths.model_dir)
            logger.info(f"Models saved to: {config.paths.model_dir}")
        
        # Step 5: Model evaluation
        logger.info("Step 5: Model evaluation...")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_multiple_models(
            trainer.trained_models, X_test, y_test
        )
        
        # Get best models
        best_model, best_model_name = trainer.get_best_model()
        best_by_precision, _ = evaluator.get_best_model_by_metric("average_precision")
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("CHURN PREDICTION PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Dataset: {Path(data_path).name}")
        logger.info(f"Total records: {df.shape[0]:,}")
        logger.info(f"Training samples: {X_train.shape[0]:,}")
        logger.info(f"Test samples: {X_test.shape[0]:,}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Class distribution - Train: {y_train.mean():.1%} churn, Test: {y_test.mean():.1%} churn")
        logger.info(f"Models evaluated: {list(trainer.trained_models.keys())}")
        
        if not args.evaluate_only:
            logger.info(f"Best model (Cross-validation): {best_model_name}")
        logger.info(f"Best model (Test Average Precision): {best_by_precision}")
        
        # Save evaluation results
        evaluator.save_evaluation_results(f"{config.paths.model_dir}/evaluation_results.pkl")
        
        # Save preprocessors
        preprocessor.save_preprocessors(config.paths.model_dir)
        
        # Print detailed evaluation report for the best model
        print("\n" + "=" * 60)
        print("DETAILED EVALUATION REPORT")
        print("=" * 60)
        evaluator.print_detailed_report(best_by_precision)
        
        # Print summary table
        print(f"\n{'MODEL PERFORMANCE SUMMARY':^60}")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'Avg Precision':<15} {'ROC AUC':<10}")
        print("-" * 60)
        
        for model_name, result in evaluation_results.items():
            accuracy = result['accuracy']
            avg_precision = result['average_precision']
            roc_auc = result['roc_auc'] or 0
            
            print(f"{model_name:<20} {accuracy:<10.4f} {avg_precision:<15.4f} {roc_auc:<10.4f}")
        
        print("-" * 60)
        
        logger.info(f"\nAll results saved to: {config.paths.model_dir}")
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()