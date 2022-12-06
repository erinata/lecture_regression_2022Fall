import pandas
from sklearn import linear_model



dataset = pandas.read_csv("dataset_regression.csv")
print(dataset)

dataset['x1sq'] = dataset['x1']**2
print(dataset)

target = dataset.iloc[:,0].values
print(target)
data = dataset.iloc[:,3:10].values
print(data)



machine = linear_model.LinearRegression()
machine.fit(data, target)



new_data = pandas.read_csv("new_dataset.csv")
new_data['x1sq'] = new_data['x1']**2
new_data = new_data.values



new_target = machine.predict(new_data)
print(new_target)


