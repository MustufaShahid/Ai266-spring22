import pandas as pd
#import file from system
train_df = pd.read_csv ('/content/train.csv')

display(train_df)

33imporint libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df=pd.read_csv('/content/train.csv')
#cleaning datasets
train_df = pd.DataFrame(train_df)
train_df['f_27'] = pd.to_numeric(train_df['f_27'], errors='coerce')
train_df = train_df.replace(np.nan, 0, regex=True)
y = train_df.target
X = train_df.drop('target', axis=1)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.4)

#navie bayes code
import math
import random
import csv
 
 filename = r'D:\user\file..csv'
 
NBmodel = csv.reader(open(filename, "rt"))
NBmodel = list(NBmodel)
NBmodel = encode_class(NBmodel)
for i in range(len(Nbmodel)):
    NBmodel[i] = [float(x) for x in NBmodel[i]]
 
     

ratio = 0.4
train_data, test_data = splitting(NBmodel, ratio)
info = MeanAndStdDevForClass(train_data)
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print("Accuracy of your model is: ", accuracy)
