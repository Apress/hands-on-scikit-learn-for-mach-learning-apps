import numpy as np
from random import randint
import matplotlib.pyplot as plt

def find_image(data, labels, d):
    for i, row in enumerate(labels):
        if d == row:
            target = row
            X_pixels = np.array(data[i])
            return (target, X_pixels)

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    target = np.load('data/mnist_targets.npy')
    print ('labels (targets):')
    print (target, br)
    print ('feature set shape:')
    print (X.shape, br)
    print ('target set shape:')
    print (y.shape, br)
    indx = randint(0, y.shape[0]-1)
    target = y[indx]
    X_pixels = np.array(X[indx])
    print ('the feature image consists of', len(X_pixels),
           'pixels')
    X_image = X_pixels.reshape(28, 28)
    plt.figure(1, figsize=(3, 3))
    title = 'image @ indx ' + str(indx) + ' is digit ' \
            + str(int(target))
    plt.title(title)
    plt.imshow(X_image, cmap='gray')
    digit = 7
    target, X_pixels = find_image(X, y, digit)
    X_image = X_pixels.reshape(28, 28)
    plt.figure(2, figsize=(3, 3))
    title = 'find first ' + str(int(target)) + ' in dataset'
    plt.title(title)
    plt.imshow(X_image, cmap='gray')
    plt.show()