from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np

df = pd.read_csv("content/combined_data.csv")

y = df["weighted_average_vote"].astype(float)
X = df.drop(
    ["imdb_title_id", "title", "weighted_average_vote"],
    axis=1,
).astype(float)


num_dimensions = X.shape[1]

# define the keras NN model
kf = KFold(n_splits=5, random_state=True, shuffle=True)
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = keras.Sequential([
        keras.layers.Dense(num_dimensions, input_dim=num_dimensions, activation="relu", kernel_initializer="normal"), #hidden
        keras.layers.Dense(1, activation="linear") # output layer
    ])

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    model.fit(X_train, y_train, epochs=10, batch_size=10)


    yhat = model.predict(X_test)
    error = mean_absolute_error(y_test, yhat)
    print("MAE: %.3f" % error)