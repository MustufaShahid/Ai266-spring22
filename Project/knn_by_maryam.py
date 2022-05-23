#knn_model accuracy
import pandas as pd 
import pickle 
import numpy as np
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/mine/Desktop/train.csv")
df.shape
del df['id']
del df['f_27'] #deleting the irrelevent column
df.head(6)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
X = df.drop(columns=['target'])
y = df['target']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

newdf = df[["f_00","f_01","f_02","f_03","f_04","f_05","f_06","f_07","f_08","f_09","f_10",
"f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_19","f_20","f_21","f_22","f_23","f_24","f_25",
"f_26","f_28","f_29","f_30","target"]]
newdf.head(6)

XTT = newdf.drop(columns=['target'])
yTT = newdf['target']
X_train, X_test, y_train, y_test = train_test_split(XTT, yTT, test_size=0.00001)
modelKNN = KNeighborsClassifier(n_neighbors=77)
resultKNN = modelKNN.fit(X_train, y_train)
prediction_test = modelKNN.predict(X_test)
accuracyKNN = metrics.accuracy_score(y_test, prediction_test)
print("KNN Accuracy: ", accuracyKNN)

dftest = pd.read_csv("C:/Users/mine/Desktop/test.csv")
dftest.head(6)

newtest = dftest[["f_00","f_01","f_02","f_03","f_04","f_05","f_06","f_07","f_08","f_09","f_10",
"f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_19","f_20","f_21","f_22","f_23","f_24","f_25",
"f_26","f_28","f_29","f_30"]]
newtest.head(6)

newCSV = dftest[['id']]
newCSV

predictionOnTest = modelKNN.predict(newtest)
predictionOnTest

newCSV['target'] = predictionOnTest 
newCSV

newCSV.to_csv('Output.csv', index=False)

print("Accuracy of your model is: ", accuracy)
