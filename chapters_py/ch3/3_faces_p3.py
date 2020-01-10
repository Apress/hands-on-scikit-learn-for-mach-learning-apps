import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import\
     LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

if __name__ == "__main__":
    br = '\n'
    warnings.filterwarnings('ignore')
    X = np.load('data/X_faces.npy')
    y = np.load('data/y_faces.npy')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    pca = PCA(n_components=0.95, whiten=True, random_state=0)
    pca.fit(X_train)
    components = pca.n_components_
    lda = LinearDiscriminantAnalysis(n_components=components)
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    svm = SVC(kernel='rbf', class_weight='balanced',
              gamma='scale', random_state=0)
    svm_name = svm.__class__.__name__
    svm.fit(X_train_lda, y_train)
    X_test_lda = lda.transform(X_test)
    y_pred = svm.predict(X_test_lda)
    cr = classification_report(y_test, y_pred)
    print ('classification report <<' + svm_name+ '>>')
    print (cr)