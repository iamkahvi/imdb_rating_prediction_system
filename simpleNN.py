from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





num_dimensions = X.shape[1]

# define the keras model

model = keras.Sequential([
    keras.layers.Dense(num_dimensions * 2, input_dim=num_dimensions, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dense(128, activation='sigmoid'), # hidden layer
    keras.layers.Dense(1, activation="linear") # output layer
])

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# model = Sequential()
# model.add(
#     Dense(num_dimensions * 2, input_dim=num_dimensions, activation="relu", kernel_initializer="he_normal")
# )
# model.add(Dense(num_dimensions, activation="sigmoid", kernel_initializer="he_normal"))
# model.add(Dense(10, activation="relu", kernel_initializer="he_normal"))
# model.add(Dense(1, activation="linear"))



model.fit(X_train, y_train, epochs=10, batch_size=10)


yhat = model.predict(X_test)
error = mean_absolute_error(y_test, yhat)
print("MAE: %.3f" % error)