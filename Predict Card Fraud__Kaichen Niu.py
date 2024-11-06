#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:06:17 2024

@author: curry
"""

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/curry/Desktop/card_fraud_data.csv")
df.head()

df.describe()

df.info()

non_frauds_percent = round(df['fraud'].value_counts()[0] / len(df) * 100, 2)
frauds_percent = round(df['fraud'].value_counts()[1] / len(df) * 100, 2)

#percentages
print('Non Frauds:', non_frauds_percent, '% of the dataset')
print('Frauds:', frauds_percent, '% of the dataset')

correlation_matrix = df.corr()

# Create a correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, linewidths=0.5)

plt.title('Correlation Heatmap of All Columns')
plt.show()

fraud_counts = df['fraud'].value_counts()
fraud_counts.index = fraud_counts.index.map({0: 'Non Fraud', 1: 'Fraud'})
plt.figure(figsize=(5, 5))
plt.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Blues'))
plt.title('Fraud vs Non Fraud Transactions')
plt.axis('equal')
plt.show()

fraud_counts = df['fraud'].value_counts()
fraud_counts.index = fraud_counts.index.map({0: 'Non Fraud', 1: 'Fraud'})
plt.figure(figsize=(5, 5))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette='Blues')
plt.title('Fraud vs Non Fraud Transactions')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()
print("Which means this data is imbalanced Dataset")

# Create a 2x2 grid of countplots
fig, axes = plt.subplots(2, 2, figsize=(8, 5))

# Plot 1: repeat_retailer vs. fraud
sns.countplot(data=df, x='repeat_retailer', hue='fraud', ax=axes[0, 0])

# Plot 2: used_chip vs. fraud
sns.countplot(data=df, x='used_chip', hue='fraud', ax=axes[0, 1])

# Plot 3: used_pin_number vs. fraud
sns.countplot(data=df, x='used_pin_number', hue='fraud', ax=axes[1, 0])

# Plot 4: online_order vs. fraud
sns.countplot(data=df, x='online_order', hue='fraud', ax=axes[1, 1])

# Set titles for each plot
axes[0, 0].set_title('Repeat Retailer vs. Fraud')
axes[0, 1].set_title('Used Chip vs. Fraud')
axes[1, 0].set_title('Used Pin Number vs. Fraud')
axes[1, 1].set_title('Online Order vs. Fraud')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Filter data by fraud category and values less than 500
fraud_data = df[(df['fraud'] == 1) & (df['distance_from_home'] < 50)]['distance_from_home']
not_fraud_data = df[(df['fraud'] == 0) & (df['distance_from_home'] < 50)]['distance_from_home']

# Create a figure with a specified size
plt.figure(figsize=(6, 4))

# Create a histogram
plt.hist(fraud_data, bins=20, alpha=0.5, color='red', label='Fraud')
plt.hist(not_fraud_data, bins=20, alpha=0.5, color='blue', label='Not Fraud')

plt.xlabel('Distance from Home')
plt.ylabel('Frequency')
plt.title('Histogram of Distance from Home (Values < 50) with Hue - Fraud')
plt.legend()

# Show the histogram
plt.show()

# Filter data by fraud category and values less than 500
fraud_data = df[(df['fraud'] == 1) & (df['distance_from_last_transaction'] < 3)]['distance_from_last_transaction']
not_fraud_data = df[(df['fraud'] == 0) & (df['distance_from_last_transaction'] < 3)]['distance_from_last_transaction']

# Create a figure with a specified size
plt.figure(figsize=(6, 4))

# Create a histogram
plt.hist(fraud_data, bins=20, alpha=0.5, color='red', label='Fraud')
plt.hist(not_fraud_data, bins=20, alpha=0.5, color='blue', label='Not Fraud')

plt.xlabel('distance_from_last_transaction')
plt.ylabel('Frequency')
plt.title('Histogram of distance_from_last_transaction (Values < 3) with Hue - Fraud')
plt.legend()

# Show the histogram
plt.show()


# Filter data by fraud category and values less than 500
fraud_data = df[(df['fraud'] == 1) & (df['ratio_to_median_purchase_price'] < 10)]['ratio_to_median_purchase_price']
not_fraud_data = df[(df['fraud'] == 0) & (df['ratio_to_median_purchase_price'] < 10)]['ratio_to_median_purchase_price']

# Create a figure with a specified size
plt.figure(figsize=(6, 4))

# Create a histogram
plt.hist(fraud_data, bins=20, alpha=0.5, color='red', label='Fraud')
plt.hist(not_fraud_data, bins=20, alpha=0.5, color='blue', label='Not Fraud')

plt.xlabel('ratio_to_median_purchase_price')
plt.ylabel('Frequency')
plt.title('Histogram of ratio_to_median_purchase_price (Values < 10) with Hue - Fraud')
plt.legend()

# Show the histogram
plt.show()

# Filter data to include only values less than 30 for all three columns
filtered_data = df[(df['distance_from_home'] < 100) & (df['distance_from_last_transaction'] < 100) & (df['ratio_to_median_purchase_price'] < 100)]

# Create a 1x3 grid of scatterplots
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Scatterplot 1: distance_from_home vs. distance_from_last_transaction
axes[0].scatter(filtered_data['distance_from_home'], filtered_data['distance_from_last_transaction'], c=filtered_data['fraud'], cmap='coolwarm', alpha=0.7, s=5)
axes[0].set_xlabel('Distance from Home')
axes[0].set_ylabel('Distance from Last Transaction')
#axes[0].set_title('Distance from Home vs. Distance from Last Transaction')

# Scatterplot 2: distance_from_home vs. ratio_to_median_purchase_price
axes[1].scatter(filtered_data['distance_from_home'], filtered_data['ratio_to_median_purchase_price'], c=filtered_data['fraud'], cmap='coolwarm', alpha=0.7, s=5)
axes[1].set_xlabel('Distance from Home')
axes[1].set_ylabel('Ratio to Median Purchase Price')
#axes[1].set_title('Distance from Home vs. Ratio to Median Purchase Price')

# Scatterplot 3: distance_from_last_transaction vs. ratio_to_median_purchase_price
axes[2].scatter(filtered_data['distance_from_last_transaction'], filtered_data['ratio_to_median_purchase_price'], c=filtered_data['fraud'], cmap='coolwarm', alpha=0.7, s=5)
axes[2].set_xlabel('Distance from Last Transaction')
axes[2].set_ylabel('Ratio to Median Purchase Price')
#axes[2].set_title('Distance from Last Transaction vs. Ratio to Median Purchase Price')

# Adjust layout
plt.tight_layout()

# Show the scatterplots
plt.show()

duplicate_count = df.duplicated().sum()
print("Count of duplicate rows:", duplicate_count)

df = df.copy()

# Selecting numerical columns to plot
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
sns.set(style="whitegrid", palette="Blues_r")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns], orient="h")
plt.title('Box Plot of Numerical Columns')
plt.xlabel('Value')
plt.ylabel('Columns')
plt.tight_layout()

plt.show()

def IQR_method(df, n, features):
    outlier_list = []
    
    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        # Appending the list of outliers 
        outlier_list.extend(outlier_list_column)
        
    outlier_list = Counter(outlier_list)        
    multiple_outliers = [k for k, v in outlier_list.items() if v > n ]
    
    print('Total number of deleted outliers:', len(multiple_outliers))
    
    return multiple_outliers

Outliers_IQR = IQR_method(df, 1, numerical_columns)

df_out = df.drop(Outliers_IQR, axis=0).reset_index(drop=True)
print('Total number after deleted outliers:', len(df_out))

# Assuming 'fraud' is your target variable
X = df_out.drop('fraud', axis=1) 
y = df_out['fraud'] 

from sklearn.model_selection import train_test_split

# Performing stratified split with a test size of 30% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

print("Train set:")
print("Total rows:", len(X_train))
print("Frauds:", y_train.sum())
print("Non Frauds:", (y_train == 0).sum())

print("\nTest set:")
print("Total rows:", len(X_test))
print("Frauds:", y_test.sum())
print("Non Frauds:", (y_test == 0).sum())

from sklearn.preprocessing import StandardScaler

# Function for scaling specific columns in a DataFrame
def Standard_Scaler(df, col_names):
    scaler = StandardScaler()
    df_scaled = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    df_scaled[col_names] = scaler.fit_transform(df_scaled[col_names])
    
    return df_scaled

# Example usage:
col_names= ['distance_from_home', 'distance_from_last_transaction',
                      'ratio_to_median_purchase_price', 'repeat_retailer', 
                      'used_chip', 'used_pin_number', 'online_order']

# Apply StandardScaler to X_train and X_test
X_train = Standard_Scaler(X_train, col_names)
X_test = Standard_Scaler(X_test, col_names)

# Print to verify
print("X_train (scaled):")
print(X_train.head())

print("\nX_test (scaled):")
print(X_test.head())

from imblearn.over_sampling import SMOTE
# Initialize SMOTE
smote = SMOTE(random_state=42)

# Perform SMOTE oversampling
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution before and after oversampling
print("Before Oversampling:", Counter(y_train))
print("After Oversampling:", Counter(y_train_resampled))

before_counts = Counter(y_train)
after_counts = Counter(y_train_resampled)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette='Blues')
plt.title('Class Distribution Before Oversampling')
plt.xlabel('Class')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=y_train_resampled, palette='Blues')
plt.title('Class Distribution After Oversampling')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on oversampled data
knn.fit(X_train_resampled, y_train_resampled)

y_pred = knn.predict(X_test)

accuracy = knn.score(X_test, y_test)
print("\nKNN Accuracy:", accuracy*100)

from sklearn.metrics import confusion_matrix, precision_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision score
precision = precision_score(y_test, y_pred)
print("\nPrecision Score:", precision)

# Plot confusion matrix with light blue palette
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
print("\nClassification Report for KNN:")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)

# Train the model on scaled oversampled data
log_reg.fit(X_train, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = log_reg.score(X_test, y_test)
print("\nLogistic Regression Accuracy:", accuracy * 100)

log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = log_reg.score(X_test, y_test)
print("\nStandard Logistic Regression Accuracy:", accuracy * 100)

# Training the Logistic Regression model with class weights
log_reg_weighted = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')
log_reg_weighted.fit(X_train, y_train)
y_pred_weighted = log_reg_weighted.predict(X_test)
accuracy_weighted = log_reg_weighted.score(X_test, y_test)
print("\nLogistic Regression with Class Weights Accuracy:", accuracy_weighted * 100)

# Evaluating and displaying results for the weighted model
cm_weighted = confusion_matrix(y_test, y_pred_weighted)
precision_weighted = precision_score(y_test, y_pred_weighted, zero_division=1)
print("\nConfusion Matrix with Class Weights:")
print(cm_weighted)
print("\nPrecision Score with Class Weights:", precision_weighted)
print("\nClassification Report with Class Weights:")
print(classification_report(y_test, y_pred_weighted))

# Plotting the confusion matrix for the weighted model
plt.figure(figsize=(10, 8))
sns.heatmap(cm_weighted, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Class Weights')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=13)

# Train the model on oversampled data
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = rf.score(X_test, y_test)
print("\nRandom Forest Accuracy:", accuracy*100)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision score
precision = precision_score(y_test, y_pred)
print("\nPrecision Score:", precision)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
print("\nClassification Report for random forest:")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

# Cross-validation for KNN
knn_cv_scores = cross_val_score(knn, X_train_resampled, y_train_resampled, cv=5)
print(f"KNN Cross-validation Scores: {knn_cv_scores}")
print(f"KNN Cross-validation Accuracy: {knn_cv_scores.mean() * 100:.2f}% ± {knn_cv_scores.std() * 100:.2f}")

# Cross-validation for Logistic Regression
log_reg_cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)
print(f"Logistic Regression Cross-validation Scores: {log_reg_cv_scores}")
print(f"Logistic Regression Cross-validation Accuracy: {log_reg_cv_scores.mean() * 100:.2f}% ± {log_reg_cv_scores.std() * 100:.2f}")

try:
    # This will fail if 'log_reg' is not fitted
    log_reg_train_accuracy = log_reg.score(X_train, y_train)
    print(f"Logistic Regression Training Accuracy: {log_reg_train_accuracy * 100:.2f}%")
except:
    # Define and fit the Logistic Regression model if not already done
    log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    log_reg.fit(X_train, y_train)  # Fit the model
    log_reg_train_accuracy = log_reg.score(X_train, y_train)  # Now score it
    print(f"Logistic Regression Training Accuracy: {log_reg_train_accuracy * 100:.2f}%")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Initialize Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)

# Training accuracy
rf_train_accuracy = rf.score(X_train, y_train)
print(f"Random Forest Training Accuracy: {rf_train_accuracy * 100:.2f}%")

# Test accuracy
rf_test_accuracy = rf.score(X_test, y_test)
print(f"Random Forest Test Accuracy: {rf_test_accuracy * 100:.2f}%")

# Cross-validation accuracy
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Random Forest Cross-validation Scores: {rf_cv_scores}")
print(f"Random Forest Cross-validation Accuracy: {rf_cv_scores.mean() * 100:.2f}% ± {rf_cv_scores.std() * 100:.2f}")

# Optionally, display the classification report
from sklearn.metrics import classification_report
y_pred_rf = rf.predict(X_test)
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Initialize models with hypothetical parameters
knn = KNeighborsClassifier(n_neighbors=3)
log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit models
knn.fit(X_train_resampled, y_train_resampled)  # Fitting KNN on resampled data for handling class imbalance
log_reg.fit(X_train, y_train)  # Fitting Logistic Regression on original training data
rf.fit(X_train, y_train)  # Fitting Random Forest on original training data

# Calculate training and test accuracies
knn_train_accuracy = knn.score(X_train_resampled, y_train_resampled)
knn_test_accuracy = knn.score(X_test, y_test)
log_reg_train_accuracy = log_reg.score(X_train, y_train)
log_reg_test_accuracy = log_reg.score(X_test, y_test)
rf_train_accuracy = rf.score(X_train, y_train)
rf_test_accuracy = rf.score(X_test, y_test)

# Perform cross-validation and compute mean and std deviation
knn_cv_scores = cross_val_score(knn, X_train_resampled, y_train_resampled, cv=5)
log_reg_cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

# Compile results into a DataFrame
results_data = {
    "Model": ["KNN", "Logistic Regression", "Random Forest"],
    "Training Accuracy": [knn_train_accuracy * 100, log_reg_train_accuracy * 100, rf_train_accuracy * 100],
    "Test Accuracy": [knn_test_accuracy * 100, log_reg_test_accuracy * 100, rf_test_accuracy * 100],
    "Cross-validation Accuracy Mean": [knn_cv_scores.mean() * 100, log_reg_cv_scores.mean() * 100, rf_cv_scores.mean() * 100],
    "Cross-validation Accuracy Std Dev": [knn_cv_scores.std() * 100, log_reg_cv_scores.std() * 100, rf_cv_scores.std() * 100]
}

results_df = pd.DataFrame(results_data)
results_df.set_index("Model", inplace=True)

# Print the DataFrame
print(results_df)

# Print classification reports for each model
print("\nClassification Report for KNN:")
print(classification_report(y_test, knn.predict(X_test)))
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, log_reg.predict(X_test)))
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))