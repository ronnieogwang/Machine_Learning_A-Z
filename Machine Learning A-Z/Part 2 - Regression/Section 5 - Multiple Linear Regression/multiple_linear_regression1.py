'''Multiple linear regression
This looks at datasets with many linear independent variabless
for simple linear, y = b0 + b1x
for multiple,       y = b0 + b1x1 + b2x2 + b3x3....bnxn
Limitations of linear regresssion
1. Linearity
2. Homoscedacity
3. multivariate normality
4. Independence of errors
5. Lack of multicollinearity
'''

#Dummy variable trap.
'''When creating dummy varibles, always eliminate one, because this can be predicted if 
others are known'''

#Building a model
'''1. All -in; throw in all your variables(prior knowledge, must have, preparing backward elimation
  
   2. Backward elimination
    *select significance level tostay in model i.e 0.05
    *fit full model with all variable
    *consider predictor with highest P-value, if P>SL, remove predictor
    *fit the model without this variable, repeat 3&4
    *if all P-values <SL, go to finish
    
   3. Forward selection
    * select significance level to enter in model i.e 0.05
    * create SLM with all varibles, select one with the lowest P-value.
    *keep this variable, add one variable of the remaining one, one a time a constrcut model,
    select one with lowest P-value. 
    *keep two variables, build a model with by adding remaining ones, @at a time. 
    If lowest P>SL, finish and select previous model. else continue.
    
    ''')

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoid dummy vriable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting linear regression model onto the training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict test set
y_pred = regressor.predict(X_test)

#plot to compare real points and predicted points
plt.plot([0,1,2,3,4,5,6,7,8,9], y_test, color = 'red')
plt.plot([0,1,2,3,4,5,6,7,8,9], y_pred, color = 'blue')
plt.title('true(red) vs predicted(blue)')
plt.xlabel('plot points')
plt.ylabel('profits')
plt.show()


