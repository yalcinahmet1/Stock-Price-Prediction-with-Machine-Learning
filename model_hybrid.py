#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:08:27 2024

@author: abdurrahmandogru
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Load the normalized dataset (replace with your actual file path)
data = pd.read_csv('/Users/abdurrahmandogru/Desktop/yap470_project/weekly_stock_data_finalized_cleaned.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns=['Date', 'Ticker', 'Label', 'Next_Week_Change'])  # Features
y = data['Label']  # Target variable: 1 (rise) or 0 (fall)

# Reshape the data for GRU (samples, time steps, features)
X = np.array(X)
X = X.reshape((X.shape[0], 2, X.shape[1] //2))  # Reshape to 3D [samples, time steps, features]

# Set timesteps and features for the input layer
timesteps = X.shape[1] # Using 1 timestep for simplicity
features = X.shape[2]  # Number of features in the dataset

# Number of repetitions
n_repeats = 10

# Prepare to store the average results of each repetition
results = {
    "Iteration": [],
    "Average Accuracy": [],
    "Average Precision": [],
    "Average Recall": [],
    "Average F1 Score": []
}

for iteration in range(1, n_repeats + 1):
    # Split data into training (80%) and testing sets (20%)
    # Split data randomly into training (80%) and testing sets (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize the hybrid model (GRU + LSTM)
    model = Sequential()
    model.add(Input(shape=(timesteps, features)))  # Define the input shape
    model.add(GRU(128, activation='tanh', return_sequences=True))  # Second GRU layer with increased neurons
    model.add(Dropout(0.1))  # Reduced dropout to retain more learning
    model.add((LSTM(128, activation='elu', return_sequences=True)))  # Add an LSTM layer with increased neurons
    model.add(Dropout(0.3))  # Reduced dropout to retain more learning
    model.add(LSTM(128, activation='sigmoid'))  # Add another LSTM layer
    model.add(Dropout(0.1))  # Reduced dropout to retain more learning
    model.add(Dense(128, activation='relu'))  # Added dense layer for better feature extraction
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    # Compile the model
    from tensorflow.keras.optimizers.legacy import Adam
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Set up 8-fold cross-validation
    kf = KFold(n_splits=8, shuffle=False)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Perform 8-fold cross-validation
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train the hybrid model on the training fold
        early_stopping = EarlyStopping(monitor='val_loss', patience=130, restore_best_weights=True, min_delta=0.01, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=100, min_lr=0.0001, verbose=0)  # Reduce learning rate when validation loss plateaus
        model.fit(X_train_fold, y_train_fold, epochs=1000, batch_size=128, validation_data=(X_val_fold, y_val_fold), verbose=0, callbacks=[early_stopping, reduce_lr])
        
        # Predict on the validation fold
        y_val_pred = (model.predict(X_val_fold) > 0.5).astype(int)  # Apply threshold for binary classification
        
        # Calculate accuracy, precision, recall, and F1 score for this fold
        accuracy_scores.append(accuracy_score(y_val_fold, y_val_pred))
        precision_scores.append(precision_score(y_val_fold, y_val_pred, zero_division=0))
        recall_scores.append(recall_score(y_val_fold, y_val_pred, zero_division=0))
        f1_scores.append(f1_score(y_val_fold, y_val_pred, zero_division=0))

    # Calculate average metrics for this iteration
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    

    # Print results for this iteration
    print(f"Iteration {iteration} - Avg Accuracy: {avg_accuracy:.2f}, Avg Precision: {avg_precision:.2f}, Avg Recall: {avg_recall:.2f}, Avg F1 Score: {avg_f1:.2f}")
    
    # Evaluate the final model on the test data (1 time)
    y_test_pred = (model.predict(X_test) > 0.5).astype(int)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Calculate and print the metrics for the test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # Store results
    results["Iteration"].append(iteration)
    results["Average Accuracy"].append(test_accuracy)
    results["Average Precision"].append(test_precision)
    results["Average Recall"].append(test_recall)
    results["Average F1 Score"].append(test_f1)
    
    print("\nTest Data Metrics:")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

# Save the results to Excel with final means included
#results_df.to_excel('cross_validation_results_with_svm.xlsx', index=False)
#print("Results saved to 'cross_validation_results_with_svm.xlsx'")

# Calculate and display the final mean of all 30 iterations
final_avg_accuracy = results_df["Average Accuracy"].mean()
final_avg_precision = results_df["Average Precision"].mean()
final_avg_recall = results_df["Average Recall"].mean()
final_avg_f1 = results_df["Average F1 Score"].mean()

# Print final means
print("\nFinal Mean Test Metrics over 30 Iterations:")
print(f"Mean Accuracy: {final_avg_accuracy:.2f}")
print(f"Mean Precision: {final_avg_precision:.2f}")
print(f"Mean Recall: {final_avg_recall:.2f}")
print(f"Mean F1 Score: {final_avg_f1:.2f}")


# Plot results with final means as horizontal lines
plt.figure(figsize=(12, 6))
plt.plot(results["Iteration"], results["Average Accuracy"], label="Average Accuracy", marker='o')
plt.plot(results["Iteration"], results["Average Precision"], label="Average Precision", marker='x')
plt.plot(results["Iteration"], results["Average Recall"], label="Average Recall", marker='^')
plt.plot(results["Iteration"], results["Average F1 Score"], label="Average F1 Score", marker='s')

# Adding table of final mean values
mean_table_data = [
    ["Mean Accuracy", f"{final_avg_accuracy:.2f}"],
    ["Mean Precision", f"{final_avg_precision:.2f}"],
    ["Mean Recall", f"{final_avg_recall:.2f}"],
    ["Mean F1 Score", f"{final_avg_f1:.2f}"]
]

table = plt.table(
    cellText=mean_table_data,
    colWidths=[0.3, 0.3],
    colLabels=["Metric", "Final Mean Values"],
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.5, 1.0, 0.3],
    rowLoc='center'
)

# Customize the plot
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.title("Average Metrics over 10 Repetitions with Hybrid Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cross_validation_results_with_svm.png")
plt.show()
