#SVR support vector regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#make x a matrix and y a vector
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

'''We will not split the databse we have a small dataset, and we need to be as accurate as possible'''
#unlike linear regressors, SVM requires feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y) #requires array
y = y.ravel()  #take it back to vector b4 fitting to model

#SVM regressor
from sklearn.svm import SVR
regressor = SVR()
regressor = SVR(kernel = 'rbf')# kernel, linear, polynomial and gaussian
regressor.fit(X, y)

#predict
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#visualization

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')# regression line
plt.title('Salary vs position')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()

