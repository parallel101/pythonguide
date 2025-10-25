import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
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


# estimator #1: if (sex = female) { if (age < 25) return 0.6; else return 0.4; } else { if (age < 15) return 0.3; else if (age > 70) return 0.2; else return 0.1; }
# estimator #2: if (age < 15) return 0.1; else return -0.1;
# estimator #3: ifafasadsad
#
# loss 误差 越小越好
#
# ResNet
# model #1: loss=0.1
# model.fit(X, y_real)
# y_residual = (model.predict(X) - y_real)
# model #2: model.fit(X, y_residual)
# y_residual_2 = (model.predict(X) - y_residual)
# model #3: model.fit(X, y_residual_2)


model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('准确率:', (y_test == y_pred).mean())

y_prob = model.predict_proba(pd.DataFrame({
    'Age': d.Age,
    'Sex': d.Sex == 'female',
    'Pclass': d.Pclass,
    'Fare': d.Fare,
    'Parch': d.Parch,
    'SibSp': d.SibSp,
    'Embarked': pd.factorize(d.Embarked)[0],
}))[:, 1] # type: ignore

for i in range(len(d)):
    print(f'乘客 {d.Name[i]:50s} 存活概率 {y_prob[i] * 100:4.2f}%  实际情况 {"存活" if d.Survived[i] else "死亡"}')
