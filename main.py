import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
url = "data/dataset.csv"  # Update this with the path to your dataset
dataset = pd.read_csv(url)

# Display the first few rows of the dataset
print(dataset.head())

# Split the dataset into features (X) and target variable (y)
X = dataset['Hours'].values.reshape(-1, 1)
y = dataset['Marks'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the training set results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Training Set - Hours vs Marks')
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.show()

# Visualize the test set results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Test Set - Hours vs Marks')
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.show()

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Predict marks for a specific number of hours
hours_to_predict = np.array([[9]])  # You can change this value
predicted_marks = model.predict(hours_to_predict)
print(f'Predicted Marks for {hours_to_predict[0][0]} hours: {predicted_marks[0]}')
