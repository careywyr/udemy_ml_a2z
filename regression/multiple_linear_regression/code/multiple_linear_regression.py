# -*- coding: utf-8 -*-
"""

@author: carey
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import numpy as np

data_path = '../data/50_Startups.csv'

dataset = pd.read_csv(data_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictiong the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
X_train = np.append(arr=np.ones((40, 1)), values=X_train, axis=1)
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()
