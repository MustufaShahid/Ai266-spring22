import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron

train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
print("TRAIN")
print(train.head())
print("TEST")
print(test.head())
del train['id']
del train['f_27']
del test['f_27']
X = train.drop(columns=['target'])
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print("Accuracy")
acc_perceptron
per_clf = Perceptron()
per_scores = cross_val_score(per_clf, X_train, y_train, cv=4)
per_mean = per_scores.mean()
print('Perceptron Accuracy after CV: ',per_mean)
newCSVTest = test[['id']]

newCSVTest

predT = test.drop(columns=['id'])
predT.head()

predictionOnTest = perceptron.predict(predT)

newCSVTest['target'] = predictionOnTest

newCSVTest.head()
newCSVTest.to_csv('Saim-PerceptronCSV.csv', index=False)

