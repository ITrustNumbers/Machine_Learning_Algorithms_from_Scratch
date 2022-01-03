import numpy as np

class MyGradientDescentRegression():

  def __init__(self, n_iterations=1000, learning_rate=0.001):

    self.weights = None #Initializing weights vector
    self.intercept = None #Initializing intercept
    self.n_iterations = n_iterations #setting number of iterations
    self.lr = learning_rate #setting learning rate

  def _Gradient_Descent(self, n_samples, X, y_preds, y_act):
    
    #Compute Gradient
    self._Dw = (-1) * (1/n_samples) * np.dot(X.transpose(), (y_act - y_preds))
    self._Di = (-1) * (1/n_samples) * np.sum(y_act - y_preds)

    #Update Model Parameters
    self.weights = self.weights - (self.lr * self._Dw)
    self.intercept = self.intercept - (self.lr * self._Di)
  
  def fit(self, X, y):

    n_samples, n_features = X.shape

    #Initializing Parameters
    self.weights = np.zeros(n_features)
    self.intercept = 0

    for _ in range(self.n_iterations):

      #Calculating Predictions
      y_preds = self.intercept + X.dot(self.weights)

      #Gradient Descent
      self._Gradient_Descent(n_samples, X, y_preds, y)

  def predict(self, x):
    if type(x) != 'numpy.ndarray':
      x = np.array(x)

    return self.intercept + x.dot(self.weights)