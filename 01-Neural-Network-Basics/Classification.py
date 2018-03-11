import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def sigmoid(z):
    return 1/(1+np.exp(-z))

def plot_sigmoid():

    sample_z = np.linspace(-10, 10, 100)
    sample_a = sigmoid(sample_z)
    plt.figure(1)
    plt.plot(sample_z, sample_a)
    plt.show()

def classify():

    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
    features = data[0]
    labels = data[1]
    plt.figure(2)
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
    plt.show()

if __name__ == "__main__":

    plot_sigmoid()
    classify()
