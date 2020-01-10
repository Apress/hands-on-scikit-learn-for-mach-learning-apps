import matplotlib.pyplot as plt, pandas as pd
from sklearn import datasets

if __name__ == "__main__":
    br = '\n'
    X, y = datasets.make_moons(n_samples=1000, shuffle=True,
                               noise=0.2, random_state=0)
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'magenta', 1:'cyan'}
    fig, ax = plt.subplots()
    data = df.groupby('label')
    for key, label in data:
        label.plot(ax=ax, kind='scatter', x='x', y='y',
                   label=key, color=colors[key])
    plt.show()