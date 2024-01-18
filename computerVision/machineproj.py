# -*- coding: utf-8 -*-
"""machineProj.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KQVC5Yi25JPI5Enu4IGJjXQ5hniU5hgU
"""

#data set link "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Modified import

# (a) Data Exploration and Preparation

# Load the dataset
train_data = pd.read_csv('/content/drive/MyDrive/mnist_train.csv')

# Data exploration
num_classes = train_data['label'].nunique()
num_features = len(train_data.columns) - 1  # Subtract 1 for the label column
print("Data Exploration:")
print("The number of the unique classes : ",num_classes)
print("The number of unique features : ", num_features)
# Check for missing values
missing_values = train_data.isnull().sum()
print("The number of missing values: ")
print(missing_values)
# split the target and features Normalize pixel values

features=train_data.drop(['label'],axis=1)
target=train_data[['label']]
normalized_features= features / 255.0

# Resize images to 28x28
image_size = (28, 28)
X = normalized_features.values.reshape(-1, *image_size, 1)

# Visualize resized 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

print("KNN USING GRID SEARCH TO GET Optimal Hyper Parameters")
# Split the training data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# (b) Experiments and Results

# Initial Experiment: K-NN Algorithm with Grid Search

# Prepare K-NN model
knn_model = KNeighborsClassifier()

# Define hyperparameters grid for grid search
param_knn = {'n_neighbors': [3, 5, 7],'weights': ['uniform', 'distance']}

# Perform grid search
search_knn = GridSearchCV(knn_model, param_grid=param_knn, cv=3)
search_knn.fit(X_train, y_train)

# Get the best K-NN model
best_knn_model = search_knn.best_estimator_

# Print the best hyperparameters for K-NN
print("Best Hyperparameters for K-NN:", search_knn.best_params_)

# Predictions on validation set using K-NN
y_pred_knn = best_knn_model.predict(X_test)

# Evaluate K-NN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-NN Validation Accuracy:", accuracy_knn)

print("---------------------------------------------------------------------------")

# Subsequent Experiment: ANN with variations in architecture
print("The ANN MODEL EXPERIMENT 1")
# Reshape data for ANN
X_train_reshaped = X_train.values.reshape(-1, *image_size, 1)
X_test_reshaped = X_test.values.reshape(-1, *image_size, 1)

# Implement and train first ANN architecture
model_ann_1 = models.Sequential([
    layers.Flatten(input_shape=image_size),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_ann_1.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_ann_1.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test))

# Predictions on validation set for the first ANN
y_pred_ann_1 = np.argmax(model_ann_1.predict(X_test_reshaped), axis=-1)


# Evaluate the first ANN model
accuracy_ann_1 = accuracy_score(y_test, y_pred_ann_1)
print("ANN Architecture 1 Validation Accuracy:", accuracy_ann_1)
print("---------------------------------------------------------------------------")
print("THE ANN EXPERIMENT 2:")
# Implement and train second ANN architecture
model_ann_2 = models.Sequential([
    layers.Flatten(input_shape=image_size),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_ann_2.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_ann_2.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test))

# Predictions on validation set for the second ANN
y_pred_ann_2 = np.argmax(model_ann_2.predict(X_test_reshaped), axis=-1)

# Evaluate the second ANN model
accuracy_ann_2 = accuracy_score(y_test, y_pred_ann_2)
print("ANN Architecture 2 Validation Accuracy:", accuracy_ann_2)

# Compare outcomes and select the best model
best_model = None
best_accu=0
y_pred_best=None
if accuracy_knn >= accuracy_ann_1 and accuracy_knn >= accuracy_ann_2:
    best_model = best_knn_model
    best_model_type = "K-NN"
    best_accu=accuracy_knn
    y_pred_best=y_pred_knn
elif accuracy_ann_1 >= accuracy_ann_2:
    best_model = model_ann_1
    best_model_type = "ANN Architecture 1"
    best_accu=accuracy_ann_1
    y_pred_best=y_pred_ann_1
else:
    best_model = model_ann_2
    best_model_type = "ANN Architecture 2"
    best_accu=accuracy_ann_2
    y_pred_best=y_pred_ann_2

print(f"The best model is {best_model_type} with Validation Accuracy: {best_accu}")

print("---------------------------------------------------------------------------")
conf_matrix = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix of the Best Model:")
print(conf_matrix)

# Save the best model to a file
if best_model_type == "K-NN":
  #this saving way because the knn don't have save function to save the model in joblib file
    joblib.dump(best_model, 'best_knn_model.pkl')
else:
  #this saving way because the ANN has function to save the joblib file
    best_model.save('best_ann_model.h5')

print(f"The model {best_model_type} is saved in a file")
print("---------------------------------------------------------------------------")


# Reload the best model from the file
if best_model_type == "K-NN":
    loaded_model = joblib.load('best_knn_model.pkl')
    print("The file best_knn_model.pkl is loaded from the file")
else:
    loaded_model = models.load_model('best_ann_model.h5')
    print("The file best_ann_model.h5 is loaded from the file")


# Load test data
test_data = pd.read_csv('/content/drive/MyDrive/mnist_test.csv')

# split the test data and Normalize pixel values for test data features

test_features=test_data.drop(['label'],axis=1)
test_target=test_data[['label']]

#normalize the test data
normalized_features_test= test_features / 255.0

# Resize images to 28x28 for test data features to match the ANN model if wanted
reshaped_features_test= normalized_features_test.values.reshape(-1, *image_size, 1)

accuracy_test=0

# Use the best model on the testing data
if best_model_type == "K-NN":
    y_test_pred = loaded_model.predict(normalized_features_test)
    accuracy_test = accuracy_score(test_target, y_test_pred)
else:
    y_test_pred = np.argmax(loaded_model.predict(reshaped_features_test), axis=-1)
    accuracy_test = accuracy_score(test_target, y_test_pred)

print("Best Model Testing Accuracy:", accuracy_test)