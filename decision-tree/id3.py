import numpy as np
import random

import pandas as pd
import sklearn as sk
import math
import pprint
import sys
import json

# Most of the code here is based on my ID3 implementation
# from assignment one. This is cited in the report.

np.random.seed(1337)
random.seed(1337)
pp = pprint.PrettyPrinter(indent=2, depth=50, width=180)


# https://stackoverflow.com/a/54405767
def is_all_same(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def entropy(df, feature_name, label_name):
    # fetch the feature and target columns
    feature = df[feature_name]
    # label = df[label_name]
    rows = len(df.index)

    # get possible values of the feature
    classes = feature.value_counts().to_dict()

    entropy = 0

    for key in classes:
        # entropy calculation. same forumula from the slides,
        # adapted for any number of feature classes instead of
        # just true/false like in the assignment
        val = classes[key]
        proportion = val / rows
        log = 0 if proportion == 0 else math.log(proportion, 2)

        entropy -= proportion * log

    return entropy

def id3(df, original, label_name, features):
    root = {}

    label = df[label_name]

    # check exit conditions
    label_all_same = is_all_same(label)
    if label_all_same:
        root["label"] = label.iloc[0]
        return root

    lowest_entropy = 999999
    best_feature = ""
    # find the lowest entropy feature
    for f in features:
        e = entropy(df, f, label_name)
        if (e < lowest_entropy):
            lowest_entropy = e
            best_feature = f

    # if there's no more features to choose from we exit here
    # with the value of the node as the mode of the target feature
    if best_feature == "":
        root["label"] = df[label_name].mode()[0]
        return root

    # split
    root["label"] = best_feature
    root["children"] = {}
    children = root["children"]

    # we need to get the classes of the feature from the whole
    # dataset, not just the subset of examples we're working with.
    # this way we can avoid some of the key-not-present errors
    # when traversing the tree later on
    classes = original[best_feature].unique()
    for c in classes:
        node = {}
        examples = df[(df[best_feature] == c)]

        if len(examples) == 0:
            # if there's no examples for this class we end here with
            # the mode of the target attribute
            node["label"] = df[label_name].mode()[0]
        else:
            new_features = features.copy()
            new_features.remove(best_feature)
            node = id3(examples, original, label_name, new_features)

        children[c] = node

    return root

# perform a tree traversal for a single sample. Returns
# the prediction for this sample, either 1 or 0
def predict_sample(sample, tree):
    curr_node = tree;
    label = curr_node["label"]
    while 'children' in curr_node:
        try:
            val = sample[label]
            child = curr_node["children"][val]
            curr_node = child
            label = curr_node["label"]
        except KeyError:
            print("validation error: key " + str(val) + " not found")
            return 0

    return label

# predicts all samples in the dataframe and returns the success rate
def predict_all(df, tree, label_name):
    correct = 0
    error = 0
    for index, row in df.iterrows():
        real = row[label_name]
        predicted = predict_sample(row, tree)
        curr_err = abs(real - predicted)
        error += curr_err

        if real == predicted:
            correct += 1

    print("prediction rate: " + str(correct / len(df) * 100) + "%")
    return error / len(df)

def tree_info(tree, features):
    label = tree["label"]
    if 'children' in tree:

        if label in features:
            features[label] += 1
        else:
            features[label] = 1

        children = tree["children"]
        max_depth = 0
        for key in children:
            (depth, features) = tree_info(children[key], features)
            if depth > max_depth:
                max_depth = depth
        return (max_depth + 1, features)
    else:
        return (0, features)

if __name__ == "__main__":
    label_name = "weighted_average_vote"

    # Read the data
    df = pd.read_csv("./content/combined_data_not_encoded.csv")
    # df = df.sample(n=1000, random_state=1337)


    # Split 70/30 randomly
    train = df.sample(frac=0.7, random_state=1337)
    test = df.drop(train.index)

    # Round the scores to the nearest 0.5 for training
    train[label_name] *= 2
    train = train.round({label_name: 0})
    print(train.head())
    print(test.head())
    train[label_name] /= 2

    # Get a list of features
    features = list(df.columns.values)
    features.remove(label_name)

    # do not consider the continuous features when building
    # the tree
    to_remove = []
    for f in features:
        num_classes = len(train[f].value_counts())
        # If the number of classes is more than 10 we consider it a continuous feature
        if num_classes > 80:
            to_remove.append(f)

    for t in to_remove:
        features.remove(t)

    # build the tree
    print("Selected features: " + str(features))
    print("Training...")
    tree = id3(train, train, label_name, features)

    # optionally pretty print the tree
    # pp.pprint(tree)

    # write the tree to json
    # with open('tree.json', 'w') as fp:
    #     json.dump(tree, fp)
    #     print("tree.json written")

    print("predicting...")
    rate = predict_all(test, tree, label_name)
    print("MAE: " + str(rate))
    # print("prediction rate: " + str(rate * 100) + "%")
    (depth, features) = tree_info(tree, {})
    print("max depth: " + str(depth))
    print("key counts in tree: ", features)
