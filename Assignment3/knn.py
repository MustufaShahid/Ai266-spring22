  #name maryam naz
  #stid:64243
  
  import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)
	from warnings import simplefilter
	simplefilter(action='ignore', category=FutureWarning)
	
	train = pd.read_csv('train.csv')
	print(train.head(5))
	
	test = pd.read_csv('test.csv')
	test.head(2)
	
	train.isnull().sum()
	train['target'] = train['target'].map({'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4})
	print(train.head(2))
	train.drop(['id'],axis=1, inplace=True)
	test.drop(['id'],axis=1, inplace=True)
	
	X = train.drop(['target'], axis=1)
	y = train['target']
	
	print(X.head(2))
	
	print(y.value_counts())
	
	from sklearn import svm
	from sklearn.model_selection import cross_val_score
	
	clf = svm.SVC(kernel='linear', C=1, random_state=42)
	clf.fit(X, y)
	
	print(clf.score(X, y))
	scores = cross_val_score(clf, X, y, cv=5)
	
	print(scores)
	
	from collections import Counter
	import numpy as np
	import matplotlib.pyplot as plt
	def euclidean_dis(x1, x2):
	return np.sqrt(np.sum((x1 - x2) ** 2))
	
	class KNN:
	def __init__(self, k=3):
	self.k = k
	
	def fit(self, X, y):
	self.X_train = X
	self.y_train = y
	
	def predict(self, X):
	y_pred = [self._predict(x) for x in X]
	return np.array(y_pred)
	
	def _predict(self, x): 
	distances = [euclidean_dis(x, x_train) for x_train in self.X_train] 
	k_idx = np.argsort(distances)[: self.k] 
	k_neighbor_labels = [self.y_train[i] for i in k_idx] 
	most_common = Counter(k_neighbor_labels).most_common(1)
	return most_common[0][0]
	if __name__ == "__main__":
	from matplotlib.colors import ListedColormap
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
	
	def accuracy(y_true, y_pred):
	accuracy = np.sum(y_true == y_pred) / len(y_true)
	return accuracy
	
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
	
	
	plt.plot(X_train, label='we are testing dataset Accuracy')
	plt.plot(y_train, label='we are training dataset Accuracy')
	plt.legend()
	plt.xlabel('X-AXIX')
	plt.ylabel('Y-AXIX')
	plt.show()
