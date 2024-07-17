# Multiple Linear Regression:-

'''
    Importing the libraries

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
    Importing the dataset

'''

dataset = pd.read_csv(r"C:\Users\ujjwa\Documents\Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

'''
    Splitting the dataset into the Training set and Test set

'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
    Training the Multiple Linear Regression model on the Training set

'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

'''
    Predicting the Test set results

'''

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # The first column in result is the 'vector of prediction' and the second column in result is the 'vector of real results' .

'''
    Evaluating the Model Performance

'''

#  The closer the 'r squared coefficient' / the result of this method 'r2_score' is to '1' , the better is the regression model is. '''
print("\n")
from sklearn.metrics import r2_score
print("Accuracy of the model: " , r2_score(y_test, y_pred) , "\n")