#simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get dataset
dataset = pd.read_csv('Salary_Data.csv')

#Note, X must be a two dimension array, else it wont work
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Here we didnot scale the data because the model we are going to use take care of that.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting values
y_pred = regressor.predict(X_test)

#Visualization
#plt.scatter(X_train, y_train, color = 'red')# real salaries
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')# regression line

plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years ofExperience')
plt.ylabel('Salary')
plt.show()
