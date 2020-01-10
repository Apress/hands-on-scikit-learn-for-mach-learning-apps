from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    br = '\n'
    digits = load_digits()
    X = digits.data
    y = digits.target
    print ('feature data shape:', X.shape)
    iso = Isomap(n_components=2)
    iso_name = iso.__class__.__name__
    iso.fit(digits.data)
    data_projected = iso.transform(X)
    print ('project data to 2D:', data_projected.shape)
    project_1, project_2 = data_projected[:, 0],\
                           data_projected[:, 1]
    plt.figure(iso_name)
    plt.scatter(project_1, project_2, c=y, edgecolor='none',
                alpha=0.5, cmap='jet')
    plt.colorbar(label='digit label', ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()