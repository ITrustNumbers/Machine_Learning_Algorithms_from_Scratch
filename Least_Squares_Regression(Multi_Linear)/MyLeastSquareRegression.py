import numpy as np
from numpy.linalg import inv

class MyLeastSquares():

  def __init__(self):
    self.coef = None #initializing a empty variable to store coefficients
    self.intercept = None #initializing a empty variable to store intercept

  def _concat_ones(self,X):
    ones = np.ones(shape = X.shape[0]).reshape(-1,1) #Creating the method to concatenate ones
    return np.concatenate((ones,X), axis = 1)

  def fit(self,X,y): #Method to fit the model to a given data
    if len(X.shape) == 1:
      X = X.reshape(-1,1)
    
    X = self._concat_ones(X)
    self.coef = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    self.intercept = self.coef[0]
    self.coef = self.coef[1:]

  def predict(self,x): #Method for prediction
    if type(x) != 'numpy.ndarray': 
      x = np.array(x)

    return self.intercept + x.dot(self.coef) #Calculate and return prediction.