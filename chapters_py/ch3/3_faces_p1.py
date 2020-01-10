import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_faces.npy')
    y = np.load('data/y_faces.npy')
    images = np.load('data/faces_images.npy')
    targets = np.load('data/faces_targets.npy')
    _, h, w = images.shape
    n_images = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(targets)
    print ('features:', n_features)
    print ('images:', n_images)
    print ('classes:', n_classes, br)
    print ('target names:')
    print (targets, br)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    pca = PCA(n_components=0.95, whiten=True, random_state=0)
    pca.fit(X_train)
    components = pca.n_components_
    eigenfaces = pca.components_.reshape((components, h, w))
    X_train_pca = pca.transform(X_train)
    pca_name = pca.__class__.__name__
    print ('<<' + pca_name + '>>')
    print ('features (after PCA):', components)
    print ('eigenface shape:', eigenfaces.shape, br)
    print (pca, br)
    svm = SVC(kernel='rbf', class_weight='balanced',
              gamma='scale', random_state=0)
    svm_name = svm.__class__.__name__
    svm.fit(X_train_pca, y_train)
    X_test_pca = pca.transform(X_test)
    y_pred = svm.predict(X_test_pca)
    cr = classification_report(y_test, y_pred)
    print ('classification report <<' + svm_name+ '>>')
    print (cr)
    ls = [np.array(eigenfaces[i].reshape(h, w))
          for i, row in enumerate(range(9))]
    fig, ax = plt.subplots(3, 3, figsize=(5, 6))
    cnt = 0
    for row in [0, 1, 2]:
        for col in [0, 1, 2]:
            ax[row, col].imshow(ls[cnt], cmap='bone',
                                aspect='auto')
            ax[row, col].set_axis_off()
            cnt += 1
    plt.tight_layout()
    plt.show()