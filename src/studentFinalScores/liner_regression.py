import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

# Features we are using
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Label we are trying to predict
predict = "G3"

# Array of just the features
X = np.array(data.drop(columns=[predict]))

# Array of just the label
y = np.array(data[predict])

# Training and testing arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):

    # Training and testing arrays
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("Studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("Studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print("Co: ", linear.coef_)
# print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

# Plotting data
p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
