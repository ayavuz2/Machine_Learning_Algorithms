import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder() # transforming non numerical into numerical data
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clss)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

print(x_train, y_test)
