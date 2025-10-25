import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# age <<
# sex > F
# pclass <<


d = sns.load_dataset('titanic')

grid = sns.FacetGrid(d, col='pclass', hue='survived')
grid.map(plt.hist, 'age', bins=4, alpha=0.5)
plt.legend()
plt.show()
