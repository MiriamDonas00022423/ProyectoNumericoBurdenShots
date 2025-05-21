import numpy as np

class LogisticRegressionGradientDescent:
    def __init__(self, n_iterations=1000, learning_rate=0.01):
        self.weights = None
        self.intercept = None
        self.n_iterations = n_iterations
        self.lr = learning_rate
        self.errors = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y_true, y_pred):
        epsilon = 1e-15  
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    def _gradient_descent(self, X, y_true, y_pred):
        n_samples = X.shape[0]
        dw = np.dot(X.T, (y_pred - y_true)) / n_samples
        db = np.sum(y_pred - y_true) / n_samples
        self.weights -= self.lr * dw
        self.intercept -= self.lr * db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.n_iterations):
            linear_output = np.dot(X, self.weights) + self.intercept
            y_pred = self._sigmoid(linear_output)
            loss = self._loss(y, y_pred)
            self.errors.append(loss)
            self._gradient_descent(X, y, y_pred)

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.intercept)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)