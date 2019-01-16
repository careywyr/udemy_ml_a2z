# -*- coding: utf-8 -*-
"""
@file    : polynomial_regression.py
@date    : 2019-01-10
@author  : carey
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np


data_path = '../data/Position_Salaries.csv'

dataset = pd.read_csv(data_path)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the Training set
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Ploynomial Regression to the Training set
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, c='r')
plt.plot(X, lin_reg.predict(X), c='b')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='r')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), c='b')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
