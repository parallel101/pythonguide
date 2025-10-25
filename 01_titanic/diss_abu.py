import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


d = pd.DataFrame({
    'score': [99, 80, 94, 40, 10, 20, 60],
    'teacher': ['Peng', 'Peng', 'Peng', 'Abu', 'Abu', 'Abu', 'Abu'],
})


sns.barplot(d, x='teacher', y='score', hue='teacher')
plt.show()
