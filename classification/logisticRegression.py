import numpy as np 

class LogisticRegression(BaseClassifier):

	def __init__(self):
		self.w, self.b = None, None

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def predict(self, X):
		predictions = []
		for x_i in range(X.shape[0]):
			proba = sigmoid(np.dot(self.w, x_i) + b)
			if proba > 0.5:
				predictions.append(1)
			else:
				predictions.append(-1)
		return np.array(predictions)

	def predict_proba(self, X):
		predictions = []
		for x_i in range(X.shape[0]):
			predictions.append(sigmoid(np.dot(self.w, x_i) + b))			
		return np.array(predictions)

