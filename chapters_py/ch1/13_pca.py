from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import\
     LinearDiscriminantAnalysis
import seaborn as sns, matplotlib.pyplot as plt

if __name__ == "__main__":
    br = '\n'
    iris = load_iris()
    X = iris.data
    y = iris.target
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    components = pca.n_components_
    model = PCA(n_components=components)
    model.fit(X)
    X_2D = model.transform(X)
    iris_df = sns.load_dataset('iris')
    iris_df['PCA1'] = X_2D[:, 0]
    iris_df['PCA2'] = X_2D[:, 1]
    print (iris_df[['PCA1', 'PCA2']].head(3), br)
    sns.set(color_codes=True)
    sns.lmplot('PCA1', 'PCA2', hue='species',
               data=iris_df, fit_reg=False)
    plt.suptitle('PCA reduction')
    lda = LinearDiscriminantAnalysis(n_components=2)
    transform_lda = lda.fit_transform(X, y)
    iris_df['LDA1'] = transform_lda[:,0]
    iris_df['LDA2'] = transform_lda[:,1]
    print (iris_df[['LDA1', 'LDA2']].head(3))
    sns.lmplot('LDA1', 'LDA2', hue='species',
               data=iris_df, fit_reg=False)
    plt.suptitle('LDA reduction')
    plt.show()