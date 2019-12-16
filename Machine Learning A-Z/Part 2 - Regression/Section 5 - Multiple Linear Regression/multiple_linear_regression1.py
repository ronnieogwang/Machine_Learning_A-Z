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
    
    '''

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

#Building an optimal model with Backward elimination
import statsmodels.api as sm
#we have to add a column of 1's to the matrix of features to cater for the b0 constant,
#the linear regression model does this on its own, but ths model doesn't
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]#optimal features, at the start, include all 0 to 5

'''OLS ordinary least squares.'''
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary() #to view p-value

#from summary, index2 has highest p value 0.814, so we remove it
X_opt = X_train[:, [0, 1, 3, 4, 5]]#optimal features, at the start, include all 0 to 5
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#from summary, index1 has highest p value 0.729, so we remove it
X_opt = X_train[:, [0, 3, 4, 5]]#optimal features, at the start, include all 0 to 5
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#from summary, index2 has highest p value 0.65, so we remove it
X_opt = X_train[:, [0, 3, 5]]#optimal features, at the start, include all 0 to 5
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#from summary, index2 has highest p value 0.07, so we remove it
X_opt = X_train[:, [0, 3]]#optimal features, at the start, include all 0 to 5
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()


'''In reference to the summary() table.
R^2 and Adjusted R^2,
to determine the best model in backward elimination, we use the set of variables which gives us the 
largest value of adjusted R^2.
In this exampe the best model would be the second last, it has adjusted R^2 = 0.947
while the last has 0.944, the second last is the best moedel.'''


'''Interpreting coefficeints
In the coefficient column,
1.If the sign is +ve, then its a positve correlation of the coefficient else -ve
2. Magnitude, is supposed to be analysed in terms of units. if the units are the same
the one with a bigger has a bigger effect.
example:
R&D : 0.7996. Means, for every unit spent on R&D, the profit increases by 79 cents

For more details, checkout the last section in the regression videos.