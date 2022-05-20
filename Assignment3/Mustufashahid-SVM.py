//mustufashahidassignment3attempt

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
train_df = pd.read_csv ('/content/drive/MyDrive/train.csv')
display(train_df)
import numpy as np
train_df = pd.DataFrame(np.random.randint(1,100, 50).reshape(-1, 1))
train_norm = train_df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
train_norm
import pandas as pd
from sklearn.model_selection import train_test_split
train_df=pd.read_csv('/content/drive/MyDrive/train.csv')
y = train_df.Cover_Type
X = train_df.drop()
(_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
test=pd.read_csv('/content/drive/MyDrive/test.csv')
test.head()
 clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Svm=clf.score(t_test,y_test)
print(" Score Of SVM",Svm*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
model_3_KNN = test[['Id']].copy()
model_3_KNN['Cover_Type'] = Cover_type
print(model_3_KNN)
model_3_KNN.to_csv('model_3_KNN.csv',index=False)

