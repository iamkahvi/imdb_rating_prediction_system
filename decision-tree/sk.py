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

possible_cols = ["year", "genre", "duration", "country", "top_actor_gender", "top_actor", "divorces", "actor_age_at_release"]
# selected_cols = ["genre", "country", "top_actor", "top_actor_gender", "language"]
# selected_cols = ["top_actor", "top_actor_gender", "divorces"]
# test_cols = selected_cols.copy()
# test_cols.append(label_name)

for col in train[possible_cols]:
    train[col] = train[col].astype("category").cat.codes

for col in test[possible_cols]:
    test[col] = test[col].astype("category").cat.codes

print(train.head())

best_mae = 9999999999
best_cols = None

for i in range(500):
    k = random.choice(range(1, len(possible_cols)))
    selected_cols = random.sample(possible_cols, k=k)
    test_cols = selected_cols.copy()
    test_cols.append(label_name)

    # Swap out these lines to change between a single decision tree
    # and a random forest
    # clf = RandomForestClassifier(n_estimators=100)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train[selected_cols], train[label_name])

    correct = 0
    error = 0
    for index, row in test.iterrows():
        predict = clf.predict(row[selected_cols].values.reshape(1, -1))
        real = row[label_name]
        curr_err = abs(real - predict[0])
        error += curr_err
        if predict[0] == real:
            correct += 1

    mae = error / 10 / len(test)
    if mae < best_mae:
        best_mae = mae
        best_cols = selected_cols.copy()

    print("MAE: " + str(mae))
    print(k, selected_cols)
    print("Rate: " + str(correct / len(test) * 100) + "%")
