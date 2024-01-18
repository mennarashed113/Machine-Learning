import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from KNN import KNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = "drug.csv"  # Make sure to provide the correct file path
data = pd.read_csv(file_path)

# preprocess

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:")
print(missing_values)

# strategy to drop rows contain missing numeric feature because the mean and median strategies are not suitable in this case
# fill with the mode in rows contain categorical feauters

# Handling numeric missing values by dropping rows
numeric_columns = data.select_dtypes(include=['number']).columns  # the only numeric feature ('Na_to_K')
data.dropna(subset=numeric_columns, inplace=True)

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Display the number of missing values after removing records
print("\nNumber of missing values after filling records:")
print(data)
print("***************************")
print(data.isnull().sum())
print("***************************")

# encode categorical columns

one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
data = one_hot_encoder.fit_transform(data[categorical_columns])

# Display the number of missing values after removing records
print("\nData after encoding:")
print(data)
print("***************************")

# Experiment 1

print("\nExperiment 1")
print("***************************")

features = data[:, :-1]  # all except last column
targets = data[:, -1]  # last column

# Initialize variables to store results
decision_tree_sizes = []
decision_tree_accuracies = []

# Repeat the experiment five times
for i in range(5):
    # Split the data into training and testing sets
    # 30% of data to test with different random splits of data
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3,
                                                        random_state=np.random.randint(1, 100))

    # Create a Decision Tree classifier
    dt_model = DecisionTreeClassifier()

    # Train the classifier on the training set
    dt_model.fit(X_train, y_train)

    # Record the size (e.g., number of nodes or depth) of the decision tree
    decision_tree_sizes.append(dt_model.tree_.node_count)

    # Make predictions on the test set
    y_pred = dt_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    decision_tree_accuracies.append(accuracy)

    # Report the size and accuracy of the decision tree
    print(f"Trail {i + 1}:")
    print(f"Size of Decision Tree: {decision_tree_sizes[i]}")
    print(f"Accuracy: {decision_tree_accuracies[i]}")
    print("\n")

# Compare the results of different models
best_model_index = np.argmax(decision_tree_accuracies)  # index of the max accuracy
best_model_accuracy = decision_tree_accuracies[best_model_index]  # value of the max accuracy
best_model_size = decision_tree_sizes[best_model_index]  # size of the max accuracy tree

# Print results
print(f"Results for the best model (Trial {best_model_index + 1} Experiment 1):")
print(f"Decision Tree Size: {best_model_size}")
print(f"Decision Tree Accuracy: {best_model_accuracy}")
print("***************************")

# clear for the next experiment
decision_tree_sizes.clear()
decision_tree_accuracies.clear()



# Experiment 2

print("\nExperiment 2")
print("***************************")

# Initialize variables to store results
experiment_results = []

# Range of training set sizes (30% to 70% in increments of 10%)
training_set_sizes = np.arange(0.3, 0.8, 0.1)

for train_size in training_set_sizes:
    mean_accuracies = []
    max_accuracies = []
    min_accuracies = []
    mean_tree_sizes = []
    max_tree_sizes = []
    min_tree_sizes = []

    for seed in range(5):  # Run the experiment five times with different random seeds
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=1 - train_size, random_state=seed)

        # Create a Decision Tree classifier
        dt_model = DecisionTreeClassifier()

        # Train the classifier on the training set
        dt_model.fit(X_train, y_train)

        # Record the size (e.g., number of nodes or depth) of the decision tree
        tree_size = dt_model.tree_.node_count

        # Make predictions on the test set
        y_pred = dt_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store results for each trial
        mean_accuracies.append(accuracy)
        mean_tree_sizes.append(tree_size)

    # Calculate mean, max, and min statistics
    mean_accuracy = np.mean(mean_accuracies)
    max_accuracy = np.max(mean_accuracies)
    min_accuracy = np.min(mean_accuracies)

    mean_tree_size = np.mean(mean_tree_sizes)
    max_tree_size = np.max(mean_tree_sizes)
    min_tree_size = np.min(mean_tree_sizes)

    # Store statistics in a report
    experiment_results.append({
        'Training Set Size': train_size,
        'Mean Accuracy': mean_accuracy,
        'Max Accuracy': max_accuracy,
        'Min Accuracy': min_accuracy,
        'Mean Tree Size': mean_tree_size,
        'Max Tree Size': max_tree_size,
        'Min Tree Size': min_tree_size
    })

# Print the experiment results
report_df = pd.DataFrame(experiment_results)
print("\nExperiment 2 Results:")
print(report_df)


# Create two plots

# Plot 1: Accuracy against Training Set Size
plt.figure(figsize=(10, 6))
plt.plot(report_df['Training Set Size'], report_df['Mean Accuracy'], label='Mean Accuracy')
plt.fill_between(report_df['Training Set Size'], report_df['Min Accuracy'], report_df['Max Accuracy'], color='gray', alpha=0.3, label='Min-Max Range')
plt.title('Accuracy vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot 2: Number of Nodes in the Final Tree against Training Set Size
plt.figure(figsize=(10, 6))
plt.plot(report_df['Training Set Size'], report_df['Mean Tree Size'], label='Mean Tree Size')
plt.fill_between(report_df['Training Set Size'], report_df['Min Tree Size'], report_df['Max Tree Size'], color='gray', alpha=0.3, label='Min-Max Range')
plt.title('Number of Nodes in the Final Tree vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Number of Nodes in the Final Tree')
plt.legend()
plt.show()