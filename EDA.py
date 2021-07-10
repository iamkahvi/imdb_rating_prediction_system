import pandas as pd
import numpy as np


dfMovies = pd.read_csv('data/IMDb movies.csv', dtype={"year": str})
dfNames = pd.read_csv('data/IMDb names.csv')
dfRatings = pd.read_csv('data/IMDb ratings.csv')
dfTitlePrincipals = pd.read_csv('data/IMDb title_principals.csv')







# print(dfMovies.shape)
# print(dfRatings.shape)
dfMovies = pd.merge(dfMovies, dfRatings, on=["imdb_title_id"])
# print(dfMovies.shape)

