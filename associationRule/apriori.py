# -*- coding: utf-8 -*-
"""
@file    : apriori.py
@date    : 2019-04-10
@author  : carey
"""

from .apyori import apriori
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Apriori on the dataset

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
myResults = [list(x) for x in results]


