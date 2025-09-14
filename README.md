# Customer Churn Prediction Pipeline

A production-ready machine learning pipeline for predicting bank customer churn using advanced feature engineering, multiple algorithms, and comprehensive model evaluation.

## Overview

This project transforms raw customer data into actionable churn predictions through a modular, scalable pipeline. The system processes customer demographics, account information, and behavioral data to identify customers likely to leave the bank, enabling proactive retention strategies.

**Key Features:**
- Automated feature engineering with statistical transformations
- Multi-algorithm comparison (Random Forest, SVM, XGBoost)
- Hyperparameter optimization using GridSearchCV
- Comprehensive evaluation with precision-recall focus for imbalanced data
- Production-ready modular architecture
- Extensive logging and error handling
- Model persistence and reusability

## Dataset

**Source:** [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

The dataset contains 10,000 customer records with 18 features including demographics (age, geography, gender), account information (balance, products, credit score), and behavioral data (activity status, complaints, satisfaction scores). The target variable indicates whether a customer has churned (1) or remained (0), with approximately 20.4% churn rate.

## Project Structure

```
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading and validation
│   │   └── validator.py        # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py         # Feature engineering pipeline
│   │   └── preprocessor.py     # Encoding, scaling, splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Model training with hyperparameter tuning
│   │   └── evaluator.py        # Model evaluation and metrics
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Logging configuration
├── notebooks/
│   └── customer_churn_analysis.ipynb  # Exploratory data analysis
├── main.py                      # Main pipeline execution
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository:**
```bash
git clone [<repository-url>](https://github.com/rab306/customer-churn-prediction-pipeline.git)
cd customer-churn-prediction
```

2. **Create virtual environment:**
```bash
python -m venv churn_venv
source churn_venv/bin/activate  # On Windows: churn_venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up data:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
   - Place the CSV file in the `data/` directory as `Customer-Churn-Records.csv`

## Usage

### Full Training Pipeline

Train all models with hyperparameter optimization:

```bash
python main.py --data_path "data/Customer-Churn-Records.csv"
```

### Evaluation Only Mode

Evaluate pre-trained models without retraining:

```bash
python main.py \
    --data_path "data/Customer-Churn-Records.csv" \
    --evaluate_only \
    --model_path "models"
```

### Configuration Options

- `--data_path`: Path to the dataset CSV file
- `--evaluate_only`: Skip training and only evaluate existing models
- `--model_path`: Path to pre-trained models (required with --evaluate_only)

## Pipeline Components

### 1. Data Processing
- **Validation**: Comprehensive data quality checks including business rule validation
- **Cleaning**: Removes unnecessary columns and handles data inconsistencies
- **Quality Reporting**: Detailed data quality metrics and validation reports

### 2. Feature Engineering
- **Age Transformation**: Box-Cox transformation to normalize age distribution (skewness: 1.01 → -0.002)
- **Balance Categorization**: Segments customers into Zero, 0-50K, and 50K+ balance groups
- **Outlier Detection**: Modified Z-score using Median Absolute Deviation
- **Feature Selection**: Removes intermediate and redundant features

### 3. Data Preprocessing
- **Encoding**: Automatic detection and encoding of categorical variables
  - Binary variables: Label encoding (Gender)
  - Multi-class variables: One-hot encoding (Geography, Card Type)
- **Scaling**: StandardScaler for numerical features
- **Splitting**: Stratified train-test split maintaining class distribution

### 4. Model Training
- **Algorithms**: Random Forest, Support Vector Machine, XGBoost
- **Hyperparameter Tuning**: GridSearchCV with 10-fold cross-validation
- **Optimization**: Focus on Average Precision score for imbalanced data
- **Class Balancing**: Weighted models to handle 20% minority class

### 5. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC, Average Precision
- **Focus**: Precision-recall analysis appropriate for churn prediction
- **Comparison**: Comprehensive model performance comparison
- **Reporting**: Detailed evaluation reports with confusion matrices

## Results

The pipeline achieves strong performance on the customer churn prediction task:

| Model | Accuracy | Average Precision | ROC AUC |
|-------|----------|-------------------|---------|
| **XGBoost** | **84.60%** | **72.13%** | **87.19%** |
| Random Forest | 85.95% | 70.18% | 86.77% |
| SVM | 85.55% | 67.88% | 84.58% |

**Best Model (XGBoost) Performance:**
- Correctly identifies 60.8% of churning customers (Recall)
- 62.6% of churn predictions are accurate (Precision)
- Strong overall discrimination ability (87.19% ROC AUC)

