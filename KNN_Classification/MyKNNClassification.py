#Creating the KNN Algorithm
import numpy as np

class MyKNNClassification:

  def __init__(self, k=3):
    
    self.k = k #Initialize the variable to hold the value of k

  def fit(self, X, y):

    self.X = X #Store the training data on memory
    self.y = y

  def _euclidean_distance(self, a, b): #Method to calculate the euclidean distance
    
    sum = 0
    for i in range(self.X.shape[1]):
      sum += (b[i] - a[i])**2

    return sum**0.5

  def _k_neighbors(self, pt): #Method ot find k nearest neighbors
    
    distances = []
    for i in range(self.X.shape[0]):
      dist = self._euclidean_distance(self.X[i], pt)
      distances.append((i, dist, self.y[i]))

      #Sorting the distances
      distances.sort(key = lambda q: q[1], reverse=False)

    return distances[0:self.k]

  def predict(self, pt):

    k_neighbors = self._k_neighbors(pt)
    vote_counts = {}
    for neighbor in k_neighbors: #Counting votes of k neighbors
      response = neighbor[2]
      vote_counts[response] = vote_counts.get(response, 0) + 1

    #Sort the votes in descending order
    Sorted_Votes = sorted(vote_counts.items(),key=lambda q: q[1], reverse=True)

    return Sorted_Votes[0][0] #return the majority vote
