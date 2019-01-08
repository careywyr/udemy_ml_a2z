# -*- coding: utf-8 -*-
"""

@author: carey
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_path = '../data/Salary_Data.csv'

dataset = pd.read_csv(data_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(X_test, y_test, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()