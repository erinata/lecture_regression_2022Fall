import pandas
from sklearn import linear_model
import numpy


dataset = pandas.read_csv("dataset_regression.csv")


target = dataset.iloc[:,2].values
print(target)
data = dataset.iloc[:,3:9].values
print(data)


machine = linear_model.LogisticRegression()
machine.fit(data, target)


new_data = pandas.read_csv("new_dataset.csv")
new_data = new_data.values

new_target = machine.predict(new_data)
print(new_target)

new_target_proba = machine.predict_proba(new_data)
numpy.set_printoptions(suppress=True)
print(new_target_proba)





