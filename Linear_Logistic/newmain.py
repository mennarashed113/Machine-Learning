import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
from sklearn.exceptions import DataConversionWarning

from sklearn.preprocessing import OneHotEncoder

matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # Importing pyplot from Matplotlib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from logistic_regression import LogisticRegression

# Load the dataset
file_path = "loan_old.csv"  # Make sure to provide the correct file path
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:")
print(missing_values)

# Check the data types of features
data_types = data.dtypes
print("\nData types:")
print(data_types)

# Check if numerical features have the same scale
numerical_data = data.select_dtypes(include=['int64', 'float64'])
scaled_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
print("\nScaled numerical data:")
print(scaled_data)
print("************************************************")

# Creating a pair plot using the scaled numerical data
sns.pairplot(scaled_data.dropna())  # Use the scaled data for the pair plot and drop rows with NaN values
plt.savefig('pairplot.png')  # Save the plot as an image

# Removing records with missing values
data_without_missing = data.dropna()

# Display the number of missing values after removing records
print("\nNumber of missing values after removing records:")
print(data_without_missing)
print("***************************")
print(data_without_missing.isnull().sum())
print("***********************************************************")

# Preprocessing
# Separate features and target columns
# separate the features from targets
features = data_without_missing.drop(['Max_Loan_Amount', 'Loan_Status'], axis=1)
targets = data_without_missing[['Max_Loan_Amount', 'Loan_Status']]

# Display the first few rows of the features and targets datasets
print("\nFeatures:")
print(features.head())

print("\nTargets:")
print(targets.head())

# Shuffle and split data into trainings and tests
features, targets = shuffle(features, targets, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# encode categorical features

categorical_features = features.select_dtypes(include=['object']).columns
categorical_features = categorical_features.drop(['Loan_ID'])  # dont encode id
categorical_features = categorical_features.tolist()

# use label encoder to avoid duplicates
encoder = LabelEncoder()

X_train_categorical = X_train[categorical_features].copy()  # Create a copy to avoid modifying the original DataFrame
X_test_categorical = X_test[categorical_features].copy()

# Apply the LabelEncoder to each column separately (because fn take only 1d array)
for column in categorical_features:
    # Fit the encoder on the training set
    X_train_categorical[column] = encoder.fit_transform(X_train_categorical[column])

    # Transform the test set using the same encoder
    X_test_categorical[column] = encoder.transform(X_test_categorical[column])

X_train_encoded = X_train_categorical
X_test_encoded = X_test_categorical

# encode categorical targets (Loan Status)
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
y_train_encoded = one_hot_encoder.fit_transform(y_train[['Loan_Status']])

# Transform the test set using the same encoder
y_test_encoded = one_hot_encoder.transform(y_test[['Loan_Status']])

# Standardize numerical features
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = StandardScaler()

standard_X_train = scaler.fit_transform(X_train[numerical_features])
# Use the same scaler to transform the test set
standard_X_test = scaler.transform(X_test[numerical_features])

# Concatenate numerical and encoded categorical features

# Loan_ID_train = X_train[['Loan_ID']].values.reshape(-1, 1) # to convert the 1D array of 'Loan_ID' into a 2D array,
X_train_final = np.concatenate((X_train_encoded, standard_X_train), axis=1)
X_test_final = np.concatenate((X_test_encoded, standard_X_test), axis=1)

# Concatenate othr targets (Max_loan) and encoded targets (Loan status )
Max_train = y_train[['Max_Loan_Amount']].values.reshape(-1, 1)
y_train_final = np.concatenate((Max_train, y_train_encoded), axis=1)

Max_test = y_test[['Max_Loan_Amount']].values.reshape(-1, 1)
y_test_final = np.concatenate((Max_test, y_test_encoded), axis=1)

# Display the shapes of the training and testing sets after encoding and standardizing
print("Training set - Features:")
print(X_train_final[:4])
print("\nTraining set - Targets:")
print(y_train_final[:10])

# Linear Regression

# Create a linear regression model
linear_model = LinearRegression()

# learn on 80% of data
# Fit the model to the training data
linear_model.fit(X_train_final, Max_train)

# predict on the other 20%
# Predict the target variable for the testing set
y_pred = linear_model.predict(X_test_final)

# cost function
# Evaluate the model using R-squared score
r2 = r2_score(Max_test, y_pred)

# Print the R-squared score
print("R-squared score:", r2)
#----------------------------------------------------------------------------------

#the logistic regression part
logistic = LogisticRegression()

logistic.fit(X_train_final, y_train_encoded)

y_pred2=logistic.predict(X_test_final).reshape((-1,1))


def accur(y_actual, y_predict):
    count= np.sum(y_actual==y_predict)
    return count/len(y_actual)

acc=accur(y_test_encoded,y_pred2)
print("The Accuracy of logistic regression")
print(acc)


#--------------------------------------------------------------------------------------------

#prediction part

new_file_path = "loan_new.csv"  # Make sure to provide the correct file path
data2 = pd.read_csv(new_file_path)
data2_without_missing = data2.dropna()

new_categorical_features = data2_without_missing.select_dtypes(include=['object']).columns
new_categorical_features = new_categorical_features.drop(['Loan_ID'])
new_categorical_features = new_categorical_features.tolist()

new_X_categorical = data2_without_missing[new_categorical_features].copy()

for column in new_categorical_features:

    new_X_categorical[column] = encoder.fit_transform(new_X_categorical[column])

X_encoded= new_X_categorical

new_numerical_features = data2_without_missing.select_dtypes(include=['int64', 'float64']).columns.tolist()
new_standard_X = scaler.fit_transform(data2_without_missing[new_numerical_features])

X_final=np.concatenate((new_standard_X,X_encoded),axis=1)

new_y_predict_linear = linear_model.predict(X_final)

new_y_predict_logistic=logistic.predict(X_final)

data2_without_missing ['Predicted_Max_Loan']= new_y_predict_linear
data2_without_missing ['Predicted_Loan_Status']= new_y_predict_logistic


#data2_without_missing .to_csv('loan_new.csv',index=False) in case we need to modify the file it self
data2_without_missing .to_csv('prediction_loan_new.csv',index=False)