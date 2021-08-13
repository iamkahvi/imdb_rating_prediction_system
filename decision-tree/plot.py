import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

data = pd.read_csv("./decision-tree/brute-force-2-uniq.txt", sep="\t", header=None)

fig, axs = plt.subplots(1, 2, figsize=(3, 5))
# axs[0, 0].hist(data[1])
a1 = axs[0]
a1.scatter(data[1], data[0])
a1.set_xlabel("Number of features used in training")
a1.set_ylabel("MAE")
# axs[0, 1].plot(data[1], data[0])
a2 = axs[1]
a2.hist2d(data[1], data[0])
a2.set_xlabel("Number of features used in training")
a2.set_ylabel("MAE")

plt.show()
