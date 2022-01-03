#Creating the Gradient Descent Logistic Regression Algorithm
import numpy as np

class MyLogisticRegression():

  def __init__(self, n_iterations=1000, learning_rate=0.001):

    self.weights = None #Initializing weights vector
    self.intercept = None #Initializing intercept
    self.n_iterations = n_iterations #setting number of iterations
    self.lr = learning_rate #setting learning rate

  def _Gradient_Descent(self, n_samples, X, y_probs, y_act):
    
    #Compute Gradient
    self._Dw = (-1) * (1/n_samples) * np.dot(X.transpose(), (y_act - y_probs))
    self._Di = (-1) * (1/n_samples) * np.sum(y_act - y_probs)

    #Update Model Parameters
    self.weights = self.weights - (self.lr * self._Dw)
    self.intercept = self.intercept - (self.lr * self._Di)

  def _sigmoid(self, y_cont): #A method for the sigmoid function

    return 1 / (1 + np.exp(-y_cont))
  
  def fit(self, X, y):

    n_samples, n_features = X.shape

    #Initializing Parameters
    self.weights = np.zeros(n_features)
    self.intercept = 0

    for _ in range(self.n_iterations):

      #Calculating Continous Prediction(Regression)
      y_cont = self.intercept + X.dot(self.weights)

      #Converting the continous prediction to class probabilities Using Sigmoid Function
      y_probs = self._sigmoid(y_cont)

      #Gradient Descent
      self._Gradient_Descent(n_samples, X, y_probs, y)

  def predict(self, x, threshold=0.5):

    if type(x) != 'numpy.ndarray':
      x = np.array(x)

    y_cont = self.intercept + x.dot(self.weights)
    y_probs = self._sigmoid(y_cont)
    y_preds = [1 if y_prob > threshold else 0 for y_prob in y_probs]
    return y_preds
