import pandas
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import metrics

dataset = pandas.read_csv("dataset_regression.csv")
dataset['x1sq'] = dataset['x1']**2
print("Dataset")
print(dataset)

target = dataset.iloc[:,2].values
# print("Target")
# print(target)
data = dataset.iloc[:,3:10].values
# print("Data")
# print(data)


data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# print("Data Test")
# print(len(data_test))
# print("Target Test")
# print(len(target_test))

# print("Data Train")
# print(len(data_train))
# print("Target Train")
# print(len(target_train))



machine = linear_model.LinearRegression()
machine.fit(data_train, target_train)

prediction = machine.predict(data_test)
# print(prediction)
# print(len(prediction))

print("R2 score for linear regression:")
print(metrics.r2_score(target_test, prediction))





target = dataset.iloc[:,2].values
# print(target)
data = dataset.iloc[:,3:9].values
# print(data)


data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)


machine = linear_model.LogisticRegression()
machine.fit(data_train, target_train)

prediction = machine.predict(data_test)

print("R2 score for logistic regression:")
print(metrics.r2_score(target_test, prediction))


