#polynomial linear regression
'''for simple linear, y = b0 + b1x
   for multiple,       y = b0 + b1x1 + b2x2 + b3x3....bnxn
   polynomial    y = b0 + b1x + b2x^2+ b3x^3+...+ bnx^n
   The polynomial is called linear wrt the coefficients'''
   
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#iloc- integer-location indexing, slicing
#make x a matrix and y a vector
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''We will not split the databse we have a small dataset, and we need to be as accurate as possible'''

#polynimial regressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#produces degree from 0 upto the specified.
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) #fit model with polynomial

#visualization
#to smoothen ou the curve, add more plotting points by adding 10 points between any two points
X_grid = np.arange(min(X), max(X), 0.1) #create vector
X_grid = X_grid.reshape((len(X_grid), 1)) #convert to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')# regression line
plt.title('Salary vs position')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()

