import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_faces.npy')
    y = np.load('data/y_faces.npy')
    targets = np.load('data/faces_targets.npy')
    print ('shape of feature and target data:')
    print (X.shape)
    print (y.shape, br)
    print ('target faces:')
    print (targets)
    X_i = np.array(X[0]).reshape(50, 37)
    image_name = targets[y[0]]
    fig, ax = plt.subplots()
    image = ax.imshow(X_i, cmap='bone')
    plt.title(image_name)
    plt.show()