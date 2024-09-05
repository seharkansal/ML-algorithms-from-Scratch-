# ML-algorithms-from-Scratch-
Machine Learning algorithm implementations from scratch.

1. linear regression-explanation
fit(self, X, y): This method trains the linear regression model using the training data X (features) and y (labels/target values).

n_samples, n_features = X.shape: Extracts the number of samples (n_samples) and features (n_features) from the shape of X.
self.weights = np.zeros(n_features): Initializes the weights as a zero vector with a length equal to the number of features.
self.bias = 0: Initializes the bias as zero.
Gradient Descent Loop:

The model is trained using gradient descent, which iteratively adjusts the weights and bias to minimize the error between the predicted values and the actual labels.
y_predicted = np.dot(X, self.weights) + self.bias: Computes the predicted values for the current iteration using the current weights and bias. This is the hypothesis function in linear regression.
dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)): Calculates the gradient of the loss function with respect to the weights. This indicates how much the weights should be adjusted.
db = (1/n_samples) * np.sum(y_predicted - y): Calculates the gradient of the loss function with respect to the bias.
self.weights -= self.lr * dw: Updates the weights by moving in the direction that minimizes the loss. The learning rate (self.lr) controls the size of the steps taken.
self.bias -= self.lr * db: Updates the bias in a similar manner.

predict(self, X): This method is used to make predictions on new data X after the model has been trained.
y_predicted = np.dot(X, self.weights) + self.bias: Calculates the predicted values using the learned weights and bias.
return y_predicted: Returns the predicted values.

2. 
