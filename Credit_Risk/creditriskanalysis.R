#=======================================================================================
#
# File:        creditriskanalysis.R
# Author:      Ali Kananitarigh
# Description: This code trains and evaluates Logistic Regression, XGBoost,
#              and Neural Network models for Exposure at Default (EAD)
#              estimation in credit risk using cross-validation and
#              hyperparameter tuning.
#
#=======================================================================================

# Clearing the R environment
rm(list=ls())

# install.packages(c("mice", "xgboost", "tidyverse", "patchwork", 
#                    "VIM", "glmnet", "corrplot", "caret", 
#                    "randomForest", "smotefamily", "nnet", 
#                    "doSNOW", "parallel"))


# Loading libraries
library(mice)
library(xgboost)
library(tidyverse)
library(patchwork)
library(VIM)
library(glmnet)
library(corrplot)
library(caret)
library(randomForest)
library(smotefamily)
library(nnet)
library(doSNOW)
library(parallel)
library(DMwR2)




#=======================================================================================
# Load Data
#=======================================================================================

data <- read.csv('/Users/alikananitarigh/Downloads/credit_risk_dataset 2.csv')
df <- data # Copy data




#=======================================================================================
# Data Exploration
#=======================================================================================

# Look at the data
glimpse(df) 
summary(df) # loan_int_rate & person_emp_length have missing values
str(df)

# Transform character variables into factors in the dataframe
df[] <- lapply(df, function(x) if(is.character(x)) as.factor(x) else x)

# Convert loan status to factor
unique(df$loan_status)
df$loan_status <- as.factor(df$loan_status)
str(df)




#=======================================================================================
# Data Visualization
#=======================================================================================

# Extract column names to begin visualization
colnames(df)

# Loan status by different ages
sort(unique(df$person_age)) 
df %>% 
    filter(person_age < 100) %>% # Exclude 123 & 144 as outliers
    select(person_age, loan_status) %>%
    ggplot(mapping = aes(x = person_age, fill = loan_status)) +
    geom_bar(alpha = 0.5) +
    theme_classic() +
    scale_x_continuous(breaks = seq(20, 75, by = 5)) +
    xlab('Age') +
    ylab('Count')

# Loan status distribution
df %>% 
    filter(person_age < 100) %>% 
    select(person_age, loan_status) %>%
    ggplot(mapping = aes(x = loan_status, fill = loan_status)) +
    geom_bar(alpha = 0.5) +
    theme_classic() + 
    xlab('Loan Status') +
    ylab('Count')

# Loan status by home ownership
df %>% 
    filter(person_age < 100) %>% 
    select(loan_status, person_home_ownership) %>%
    ggplot(mapping = aes(x = person_home_ownership, fill = loan_status)) +
    geom_bar(alpha = 0.5) +
    theme_classic() + 
    xlab('Home Ownership') +
    ylab('Count')

# Loan status by income
df %>% 
    filter(person_income < 200000) %>% 
    select(person_income, loan_status) %>%
    ggplot(mapping = aes(x = person_income, fill = loan_status)) +
    geom_density(alpha = 0.5) +
    theme_classic() + 
    facet_wrap(~loan_status) + 
    xlab('Annual Income') + 
    ylab('Density')

# Loan status by employment length
sort(unique(df$person_emp_length)) # Min: 0 , Max: 123 (outlier)
df %>% 
    filter(complete.cases(person_emp_length) & person_emp_length <= 47) %>% # 65 - 18 = 47
    ggplot(mapping = aes(x = person_emp_length, fill = loan_status)) +
    geom_bar(alpha = 0.5) +  
    theme_classic() + 
    facet_wrap(~ loan_status) +
    xlab('Years of Employment') + 
    ylab('Count')

# Employment length distribution
df %>% 
    filter(complete.cases(person_emp_length) & person_emp_length <= 47) %>% # 65 - 18 = 47 (assuming work starts at age 18)
    ggplot(mapping = aes(x = person_emp_length, fill = loan_status)) +
    geom_bar(alpha = 0.5) +  
    theme_classic() + 
    xlab('Years of Employment') + 
    ylab('Count')

# Loan status by loan intent
df %>% 
    filter(complete.cases(loan_intent)) %>% 
    ggplot(mapping = aes(x = loan_status, fill = loan_intent)) +
    geom_bar(alpha = 0.5) +
    facet_wrap(~ loan_intent) +
    theme_classic() + 
    xlab('Loan Status') + 
    ylab('Count') 

# Loan status by loan grade
df %>% 
    ggplot(mapping = aes(x = loan_grade, fill = loan_status)) +  
    geom_bar(alpha = 0.5, position = "dodge") + 
    theme_classic() + 
    xlab('Loan Grade') + 
    ylab('Count')

# Loan status by loan intent and grade
df %>% 
    ggplot(mapping = aes(x = loan_grade, fill = loan_status)) +  
    geom_bar(alpha = 0.5, position = "dodge") + 
    facet_wrap(~ loan_intent) + 
    theme_classic() + 
    xlab('Loan Grade') + 
    ylab('Count')

# Loan status by home ownership and grade
df %>% 
    ggplot(mapping = aes(x = loan_grade, fill = loan_status)) +  
    geom_bar(alpha = 0.5, position = "dodge") +  
    facet_wrap(~ person_home_ownership) + 
    theme_classic() + 
    xlab('Loan Grade') + 
    ylab('Count')

# Loan amount by loan intent, loan grade, and home ownership ########
# First plot: Loan amount by loan intent
plot1 <- df %>% 
    filter(complete.cases(loan_amnt, loan_status)) %>%  
    ggplot(mapping = aes(x = loan_intent, y = loan_amnt, fill = loan_status)) +  
    geom_boxplot(alpha = 0.5) +  
    theme_classic() +
    xlab('Loan Intent') +
    ylab('Loan Amount') +  
    scale_fill_brewer(palette = "Set2") +  
    labs(title = "Loan Amount by Loan Intent") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Second plot: Loan amount by home ownership
plot2 <- df %>% 
    filter(complete.cases(loan_amnt)) %>%  
    ggplot(mapping = aes(x = person_home_ownership, y = loan_amnt, fill = loan_status)) +  
    geom_boxplot(alpha = 0.5) +  
    theme_classic() +
    xlab('Home Ownership') +
    ylab('Loan Amount') +  
    scale_fill_brewer(palette = "Set2") +  
    labs(title = "Loan Amount by Home Ownership")

# Third plot: Loan amount by loan grade
plot3 <- df %>% 
    filter(complete.cases(loan_amnt)) %>%  
    ggplot(mapping = aes(x = loan_grade, y = loan_amnt, fill = loan_status)) +  
    geom_boxplot(alpha = 0.5) +  
    theme_classic() +
    xlab('Loan Grade') +
    ylab('Loan Amount') +  
    scale_fill_brewer(palette = "Set2") +  
    labs(title = "Loan Amount by Loan Grade")

# Combine all three plots into a grid
combined_plot <- (plot1 / plot2 / plot3)  # Stacked vertically
combined_plot




#=======================================================================================
# Feature Engineering
#=======================================================================================
# In this section, we perform: 
# - Imputation
# - Feature creation 
# - Feature selection
# - Encoding
# - Outlier detection
# - Scaling

#############################################################################
# Imputation
# Checking for missing values

# Count missing values in each column
colSums(is.na(df))

# Percentage of missing values in the dataset
sum(is.na(df)) / (nrow(df) * ncol(df)) # 1% of the data is missing

# Visualizing missing data patterns
md.pattern(df)
aggr(df)

# Creating new features to indicate missing values (these may follow a pattern)
df$missing_emp_length <- ifelse(is.na(df$person_emp_length), "Y", "N")
df$missing_loan_int <- ifelse(is.na(df$loan_int_rate), "Y", "N")
df$missing_emp_length <- as.factor(df$missing_emp_length)
df$missing_loan_int <- as.factor(df$missing_loan_int)  # Fixed incorrect assignment
str(df)

# Validating loan_percent_income with manual calculation
nrow(df) - sum(signif(df$loan_amnt / df$person_income, 2) == df$loan_percent_income) # Expected = 9045
df$loan_to_income <- df$loan_amnt / df$person_income

#############################################################################
# Feature Selection

# Displaying feature names
colnames(df)

# Dropping 'loan_percent_income' from the dataset
features <- c("loan_status", "person_age", "person_income", "person_home_ownership", 
              "person_emp_length", "loan_intent", "loan_grade",
              "loan_amnt", "loan_int_rate", 
              "loan_to_income", "cb_person_default_on_file",
              "cb_person_cred_hist_length", "missing_emp_length",
              "missing_loan_int")

df <- df[, features]
str(df)

#############################################################################
# Encoding

# Creating dummy variables for categorical features
dummies <- dummyVars(~ ., data = select(df, -loan_status))
dummies.df <- as.data.frame(predict(dummies, select(df, -loan_status)))
View(dummies.df)

#############################################################################
# Imputation using Bagging Method

# Impute missing values using the bagging method
set.seed(123)  # Ensures reproducibility in bagging imputation
pre.process <- preProcess(dummies.df, method = 'bagImpute')
imputed.df <- as.data.frame(predict(pre.process, dummies.df))
View(imputed.df)

# Replacing original columns with imputed values
df$person_emp_length <- imputed.df$person_emp_length
df$loan_int_rate <- imputed.df$loan_int_rate

# Checking missing data patterns after imputation
md.pattern(df)
aggr(df)

#############################################################################
# Outlier Detection & Removal

# Checking outliers in employment length
boxplot(df$person_emp_length) 
# Removing entries where employment length is greater than 47 years
df <- df[df$person_emp_length <= (65 - 18), ] # 18: Starting age, 65: Retirement age
boxplot(df$person_emp_length) 

# Checking and removing outliers in age
boxplot(df$person_age)
df <- df[df$person_age <= 80, ]
boxplot(df$person_age)

# Checking and removing outliers in income
boxplot(df$person_income)
df <- df[df$person_income <= 1000000, ] # Removing incomes above 1 million
boxplot(df$person_income)

# Checking loan amount distribution
boxplot(df$loan_amnt)

#############################################################################
# Scaling

# Standardizing numerical variables
df$person_age <- scale(df$person_age)
df$person_emp_length <- scale(df$person_emp_length)
df$cb_person_cred_hist_length <- scale(df$cb_person_cred_hist_length)
df$loan_int_rate <- scale(df$loan_int_rate)
df$loan_to_income <- scale(df$loan_to_income)
df$loan_amnt <- scale(df$loan_amnt)
df$person_income <- scale(df$person_income)
str(df)

#############################################################################
# Data Splitting

# Splitting dataset into training and testing sets (70% training, 30% testing)
set.seed(123)
indexes <- createDataPartition(df$loan_status, times = 1, p = 0.7, list = FALSE)
df.train <- df[indexes,]
df.test <- df[-indexes,]

# Checking class distribution in the original, training, and test sets
prop.table(table(df$loan_status))
prop.table(table(df.train$loan_status))
prop.table(table(df.test$loan_status))

#############################################################################
# Correlation Analysis

# Computing and visualizing correlation among numeric features
numeric_cols <- sapply(df, is.numeric)
corrplot(cor(df[, numeric_cols]))




#=======================================================================================
# Model Training and Evaluation 
#=======================================================================================

# Set seed for reproducibility in model training & cross-validation
set.seed(123)  

# 5-fold cross-validation with 3 repeats
trn.cntrl <- trainControl(method = "repeatedcv", number = 5, 
                          repeats = 3,
                          search = "grid")

# Parallel Processing Setup for Hyperparameter Tuning
n_cores <- detectCores(logical = TRUE) - 1
cl <- makeCluster(n_cores, type = "SOCK") 
registerDoSNOW(cl)  


######################### Logistic Regression Model ########################

# Define the hyperparameter grid for tuning logistic regression
tune.grid.lr <- expand.grid(
    alpha = c(0, 0.5, 1),
    lambda = c(0.001, 0.01, 0.1, 1, 10, 100)
)

View(tune.grid.lr)

# Train the logistic regression model using cross-validation
caret.lr.cv <- train(loan_status ~ ., 
                     data = df.train, 
                     method = "glmnet", 
                     family = "binomial",
                     tuneGrid = tune.grid.lr, 
                     trControl = trn.cntrl)

# Display trained model details
caret.lr.cv

# Train the final logistic regression model using the best hyperparameters
final_model <- train(loan_status ~ ., 
                     data = df.train, 
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = caret.lr.cv$bestTune$alpha,
                                            lambda = caret.lr.cv$bestTune$lambda), 
                     trControl = trn.cntrl)

# Make predictions on the test set
pred.lr <- predict(final_model, newdata = df.test)

# Evaluate model performance with a confusion matrix
cm.lr <- confusionMatrix(pred.lr, df.test$loan_status)

##################### eXtreme Gradient Boosting Model ####################

# Define the hyperparameter grid for tuning XGBoost
tune.grid.xgb <- expand.grid(eta = c(0.05, 0.075, 0.1),
                             nrounds = c(50, 100),
                             max_depth = 5:7,
                             min_child_weight = c(2, 2.25, 2.5),
                             colsample_bytree = c(0.3, 0.4, 0.5),
                             gamma = 0,
                             subsample = 1)

View(tune.grid.xgb)

# Train the XGBoost model using cross-validation
caret.xgb.cv <- train(loan_status ~ ., 
                      data = df.train, 
                      method = "xgbTree", 
                      tuneGrid = tune.grid.xgb, 
                      trControl = trn.cntrl)

# Display trained model details
caret.xgb.cv

# Make predictions on the test set using the best number of boosting rounds
pred.xgb <- predict(
    caret.xgb.cv, 
    newdata = df.test, 
    iteration_range = c(1, caret.xgb.cv$bestTune$nrounds)
)

# Evaluate model performance with a confusion matrix
cm.xgb <- confusionMatrix(pred.xgb, df.test$loan_status)

########################### Neural Networks ############################

# Define the hyperparameter grid for tuning the neural network
tune.grid.nnet <- expand.grid(
    size = c(5, 10, 15, 20, 25, 30),  
    decay = c(0, 0.001, 0.01, 0.05, 0.1, 0.5, 1)
)

View(tune.grid.nnet)

# Train the neural network model using cross-validation
caret.nnet.cv <- train(
    loan_status ~ ., 
    data = df.train, 
    method = "nnet", 
    tuneGrid = tune.grid.nnet, 
    trControl = trn.cntrl,
    trace = FALSE,
    maxit = 200
)

# Make predictions on the test set
pred.nnet <- predict(caret.nnet.cv, newdata = df.test)

# Evaluate model performance with a confusion matrix
cm.nn <- confusionMatrix(pred.nnet, df.test$loan_status)




#=======================================================================================
# Balancing loan_status
#=======================================================================================

# Check for imbalance in the target variable
prop.table(table(df$loan_status))
prop.table(table(df.train$loan_status))
prop.table(table(df.test$loan_status))

# Create dummy variables for the training set to use SMOTE
dummies <- dummyVars(~ ., data = select(df.train, -loan_status))
dummies.df.train <- as.data.frame(predict(dummies, select(df.train, -loan_status)))
dummies.df.train$loan_status <- as.factor(df.train$loan_status)

# Apply ROSE (Random Over-Sampling Examples) to balance the data
#df.train <- ROSE(loan_status ~ ., data = dummies.df.train, seed = 123)$data

# Apply SMOTE (Synthetic Minority Over-sampling Technique) for balancing
set.seed(123)  # Set seed for SMOTE balancing (ensures the same synthetic data)
df.train.smote <- SMOTE(X = dummies.df.train[, -which(names(dummies.df.train) == "loan_status")],  
                  target = dummies.df.train$loan_status, 
                  K = 5, 
                  dup_size = 2)$data

# Rename the synthetic class column and remove the old one
df.train.smote$loan_status <- as.factor(df.train.smote$class)
df.train.smote <- df.train.smote[, -which(names(df.train.smote) == "class")]

# Verify the distribution of loan_status after balancing
table(df.train.smote$loan_status)

# Create dummy variables for the test set to match the training set format
dummies <- dummyVars(~ ., data = select(df.test, -loan_status))
dummies.df.test <- as.data.frame(predict(dummies, select(df.test, -loan_status)))
dummies.df.test$loan_status <- as.factor(df.test$loan_status)




#=======================================================================================
# Model Training and Evaluation (after Balancing)
#=======================================================================================

# Set seed for reproducibility in model training & cross-validation
set.seed(123)  

# Define 5-fold cross-validation with 3 repeats
trn.cntrl <- trainControl(method = "repeatedcv", number = 5,
                          repeats = 3,
                          search = "grid")

######################## Logistic Regression Model ########################

# Define the hyperparameter grid for tuning logistic regression
tune.grid.lr <- expand.grid(
    alpha = c(0, 0.5, 1),
    lambda = c(0.001, 0.01, 0.1, 1, 10, 100)
)

View(tune.grid.lr)

# Train the logistic regression model using cross-validation on SMOTE-balanced training data
caret.lr.cv.blnc <- train(loan_status ~ ., 
                     data = df.train.smote,
                     method = "glmnet", 
                     family = "binomial",
                     tuneGrid = tune.grid.lr, 
                     trControl = trn.cntrl)

# Display trained model details
caret.lr.cv.blnc

# Train the final logistic regression model using the best hyperparameters
final_model_lr_blnc <- train(loan_status ~ ., 
                        data = df.train.smote,
                        method = "glmnet",
                        tuneGrid = expand.grid(alpha = caret.lr.cv.blnc$bestTune$alpha,
                                               lambda = caret.lr.cv.blnc$bestTune$lambda), 
                        trControl = trn.cntrl)

# Make predictions on the dummy-encoded test set
pred.lr.blnc <- predict(final_model_lr_blnc, newdata = dummies.df.test)

# Evaluate model performance with a confusion matrix
cm.lr.blnc <- confusionMatrix(pred.lr.blnc, dummies.df.test$loan_status)

##################### eXtreme Gradient Boosting Model ####################

# Define the hyperparameter grid for XGBoost
tune.grid.xgb <- expand.grid(
    eta = c(0.05, 0.075, 0.1),
    nrounds = c(50, 100),
    max_depth = 5:7,
    min_child_weight = c(2, 2.25, 2.5),
    colsample_bytree = c(0.3, 0.4, 0.5),
    gamma = 0,
    subsample = 1)

View(tune.grid.xgb)

# Train the XGBoost model using cross-validation on SMOTE-balanced training data
caret.xgb.cv.blnc <- train(loan_status ~ .,
                      data = df.train.smote,
                      method = "xgbTree",
                      tuneGrid = tune.grid.xgb,
                      trControl = trn.cntrl)

# Display trained model details
caret.xgb.cv.blnc

# Make predictions on the test dataset
pred.xgb.blnc <- predict(
    caret.xgb.cv.blnc,
    newdata = dummies.df.test,
    iteration_range = c(1, caret.xgb.cv.blnc$bestTune$nrounds)
)

# Evaluate model performance with a confusion matrix
cm.xgb.blnc <- confusionMatrix(pred.xgb.blnc, dummies.df.test$loan_status)

########################### Neural Network Model ############################

# Define the hyperparameter grid for Neural Networks
tune.grid.nnet <- expand.grid(
    size = c(5, 10, 15, 20, 25),
    decay = c(0, 0.001, 0.01, 0.1, 1) 
)

View(tune.grid.nnet)

# Train the Neural Network model using using cross-validation on SMOTE-balanced training data
caret.nnet.cv.blnc <- train(
    loan_status ~ .,
    data = df.train.smote,
    method = "nnet",
    tuneGrid = tune.grid.nnet,
    trControl = trn.cntrl,
    trace = FALSE,
    maxit = 200
)

# Make predictions on the test dataset
pred.nnet.blnc <- predict(caret.nnet.cv.blnc, newdata = dummies.df.test)

# Evaluate model performance with a confusion matrix
cm.nn.blnc <- confusionMatrix(pred.nnet.blnc, dummies.df.test$loan_status)

# Stop parallel processing and reset
stopCluster(cl)
registerDoSEQ()