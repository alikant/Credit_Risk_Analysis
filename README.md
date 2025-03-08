# Credit Risk Analysis - Exposure at Default (EAD) Prediction

## Overview
This repository contains code for predicting Exposure at Default (EAD) in credit risk assessment.

## Prerequisites
The following R packages are required:
```R
install.packages(c("mice", "xgboost", "tidyverse", "patchwork", 
                  "VIM", "glmnet", "corrplot", "caret", 
                  "randomForest", "smotefamily", "nnet", 
                  "doSNOW", "parallel", "DMwR2"))
```

## Dataset
The code makes use of a credit risk dataset that contains a wide range of personal and loan specific features. The dataset is available in this repository.

## Implementation

### Data Preprocessing
1. **Data Exploration and Visualization**
   - Performing descriptive analysis
   - Exploring relationships between the response variable (‘loan status’) and features using visualizations

2. **Feature Engineering**
   - Imputing missing values, creating indicators, selecting relevant features, encoding categorical variables, detecting outliers, and standardizing numerical data
   
3. **Data Splitting**
   - Stratified sampling – 70% and 30% split for training and testing, respectively

### Model Training
- Models trained with Logistic regression, XGBoost, and Neural networks
- Hyperparameter tuning

### Class Imbalance Handling
- Applying SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes
- Retraining models based on balanced data

## Results
- Comparison of models trained on imbalanced vs. balanced data
- Using balanced data improves accuracy

## Future Improvement
- Feature importance analysis
- Investigating other classification models, like Random Forest and SVM
- Using ensemble techniques
- Applying different methods to balance data
