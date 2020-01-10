import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

if __name__ == "__main__":
    br = '\n'
    digits = load_digits()
    print (digits.keys(), br)
    print ('2D shape of digits data:', digits.images.shape, br)
    X = digits.data
    y = digits.target
    print ('X shape (8x8 flattened to 64 pixels):', end=' ')
    print (X.shape)
    print ('y shape:', end=' ')
    print (y.shape, br)
    i = 500
    print ('vector (flattened matrix) of "feature" image:')
    print (X[i], br)
    print ('matrix (transformed vector) of a "feature" image:')
    X_i = np.array(X[i]).reshape(8, 8)
    print (X_i, br)
    print ('target:', y[i], br)
    print ('original "digits" image matrix:')
    print (digits.images[i])
    plt.figure(1, figsize=(3, 3))
    plt.title('reshaped flattened vector')
    plt.imshow(X_i, cmap='gray', interpolation='gaussian')
    plt.figure(2, figsize=(3, 3))
    plt.title('original images dataset')
    plt.imshow(digits.images[i], cmap='gray',
               interpolation='gaussian')
    plt.show()