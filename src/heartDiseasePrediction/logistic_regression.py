import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import style
import seaborn as sns


column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "diagnosis"
]

data = pd.read_csv("processed.cleveland.data", header=None, names=column_names, na_values=["?"])

for column in data.columns:
    if data[column].dtype != object:
        data[column] = data[column].fillna(data[column].median())

data = data[["age", "trestbps", "chol", "fbs", "thalach", "oldpeak", "diagnosis"]]
data["diagnosis"] = data["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
predict = "diagnosis"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

#best = 0
#for _ in range(3000):
#
#    # Training and testing arrays
#    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
#
#    logistic = linear_model.LogisticRegression(max_iter=3000, solver="liblinear")
#
#    logistic.fit(x_train, y_train)
#
#    acc = logistic.score(x_test, y_test)
#
#    if acc > best:
#        best = acc
#        with open("HeartDiseaseModel.pickle", "wb") as f:
#            pickle.dump(logistic, f)
#
#print(f"Best : {best}")

pickle_in = open("HeartDiseaseModel.pickle", "rb")
logistic = pickle.load(pickle_in)

predictions = logistic.predict(x_test)

#for x in range(len(predictions)):
#    print(predictions[x], y_test[x])

plt.figure(figsize=(10, 8))

corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Heart Disease Features')
plt.tight_layout()
plt.show()
