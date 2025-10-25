import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
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


y_train = X_train.pop('Survived') # type: ignore
y_test = X_test.pop('Survived') # type: ignore


# 800 -> 240 survived , 560 dead
#  weight  400 / 240  , 400 / 560


# if (sex = female) { if (age < 25) return 1; else return 0; } else { if (age < 15) return 1; else if (age > 70) return 1; else return 0; }


model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print((y_test == y_pred).mean())
y_pred = model.predict(X_train)
print((y_train == y_pred).mean())
