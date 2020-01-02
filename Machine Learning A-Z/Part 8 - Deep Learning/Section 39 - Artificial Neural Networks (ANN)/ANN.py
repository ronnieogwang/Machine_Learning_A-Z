'''
**The Neuron ***
**Activation function**
    1. Threshold function; 1 if X>= 0 else 0 where x is the weighted sum
    2. Sigmoid, returns probabilty btn 0 and 1: 1/(1+e^-x)
    3. Rectifier,  0 for x<0, linear after that
    4. Hyperbolic function, is a sigmoid with negative values
    5. softmax, is the sigmoid applied to more than two variables
    
**Cost function**
Measures the error in the prediction. The lower the cost fucntion, the more accurate the prediction

**stepwise**
1. Randomly intialise the weights
2. Input each observation, each feature isa node
3. Forward propagate and get prediction
4. Compare with actual result and measure error
5. Back propagate and adjust weights
6. repeat 2 to 5 for @observation (reinforcement learning) or after a batch (batch learning)
 
**Theano Library**
Numerical computation library, very fast computation, uses numpy syntax.
pros; not only can it run on our cpu but also gpu
we need gpu's due to their parallel processing architecture that is x-tic of ANN's

**Tensor flow**
'''
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()#object for first categoriacal feature
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()#object for second categoriacal feature
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]#removes the first column to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, we have to do this as a must in ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential#initalize ann
from keras.layers import Dense #create layers

# Initialising the ANN, defining the ann as a sequence of layers
classifier = Sequential()

# Adding the input layer and the first hidden layer
#the dense function intializes the weights to small values close to zero
#number of node = number of independent variables
#neurons are use an activation function to determine whether to pass a strong signal or not
#output_dim; nodes in hidden layer, rule of thumb use the average of I/P & O/P nodes
#init=uniform, initialize weights with uniform distribution
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN, add stochstic gradient to the ann
#loss function, function on which optimizer is applied
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#batch size, number of observations before updating weights
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)