import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# age << 0 ~ 90
# sex > F , M
# pclass << 1, 2, 3
# survived = 1, 0


# 800 -> 600 train 200 test


d = pd.read_csv('dataset/Titanic-Dataset.csv')
print(d.columns)
print(d)


X = pd.DataFrame({
    'Age': d.Age,             # float
    'Sex': d.Sex == 'female', # bool -> True 1, False 0
    'Pclass': d.Pclass,       # int: 3
    'Fare': d.Fare,           # float
    'Parch': d.Parch,         # int
    'SibSp': d.SibSp,         # int
    'Embarked': pd.factorize(d.Embarked)[0],  # int: 0=C, 1=S, 2=Q, -1=NaN
    'Survived': d.Survived,   # int: 0, 1
})
X = X.dropna()

print(X.corr())

pivot = int(len(X) * 0.8)
X_train = X[:pivot]
X_test = X[pivot:]

X_train, X_test = train_test_split(X, test_size=0.2, shuffle=True, random_state=42)


y_train = X_train.pop('Survived')
y_test = X_test.pop('Survived')


model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print((y_test == y_pred).mean())
