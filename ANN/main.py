import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('C:/Users/ANKITA ADITYA/Desktop/Data sets/churn_modelling.csv')
#input features
X = dataset.iloc[:, 3:13].values
#output
y = dataset.iloc[:, 13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder (categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#feature scaling to ease complex calculation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#---------------data preprocessing done----------------|

#ANN model build up
import keras
from tensorflow.keras import Sequential
from keras.layers import Dense
# these two to create layes in ANN

#intializing ANN
classifier = Sequential()

#adding i/p and first hidden layer
#output_dim = no. of i/p +o/p layers / 2 = hidden layers = 6
#init to randomly assign weights to the nodes
classifier.add(Dense(6, kernel_initializer = 'uniform', activation='relu', input_dim=11))

#adding another hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation='relu'))

#output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))
#use softmax if you have more than one dependent variable

#compiling the ANN
#adam is the stochastic gradient descent most commomly used form
#loss = categorical_crossentropy for more than 2 dependent variables
#metrics = takes list
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

#fitting ANN to training set
#first two parameters are independent and dependent variables
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#predicting the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # just same as if statement.. will give result as true or false

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
#this checks the no. of correct and incorrect predictions
#no.of correct +incorrect prediction / total no. of test set = accuracy

#----------DONE---------------|

