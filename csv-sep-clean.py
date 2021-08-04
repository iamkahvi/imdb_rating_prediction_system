import pandas as pd

df = pd.read_csv("./content/names.csv")
print(df.head())
print(df.shape)
df.to_csv("./content/names-sep-new.csv", index=False, sep=";")

df = pd.read_csv("./content/movies.csv")
print(df.head())
print(df.shape)
df.to_csv("./content/movies-sep-new.csv", index=False, sep=";")
