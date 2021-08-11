import numpy as np
import random

import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import math
import pprint
import sys
import json

label_name = "weighted_average_vote"

np.random.seed(1337)
random.seed(1337)

df = pd.read_csv("./content/combined_data_not_encoded.csv");

# df = df.sample(n=1000, random_state=1337)
df[label_name] *= 10

train=df.sample(frac=0.7, random_state=1337)
test=df.drop(train.index)

print(train.head())

train_cols = ["genre", "country", "top_actor", "top_actor_gender", "language"]
# train_cols = ["top_actor", "top_actor_gender", "divorces"]
test_cols = train_cols.copy()
test_cols.append(label_name)

for col in train[train_cols]:
    train[col] = train[col].astype("category").cat.codes

for col in test[train_cols]:
    test[col] = test[col].astype("category").cat.codes

print(train.head())

# Swap out these lines to change between a single decision tree
# and a random forest

# clf = RandomForestClassifier(n_estimators=50)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train[train_cols], train[label_name])

correct = 0
error = 0
for index, row in test.iterrows():
    predict = clf.predict(row[train_cols].values.reshape(1, -1))
    real = row[label_name]
    curr_err = abs(real - predict[0])
    error += curr_err
    if predict[0] == real:
        correct += 1

print("MAE: " + str(error / 10 / len(test)))
print("Rate: " + str(correct / len(test) * 100) + "%")

# tree.plot_tree(clf)
