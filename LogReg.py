import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
class LogReg:
    def __init__(self, learningRate=0.01, iterations=1000):
        self.lr = learningRate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.costHistory = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def train(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.costHistory.append(self.cost(h, y))

    def test(self, X):
        return (self.sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5).astype(int)
    
    def costPlot(self):
        plt.plot(self.costHistory)
        plt.title("Cost Function Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()