import numpy as np, pandas as pd, seaborn as sns

if __name__ == "__main__":
    br = '\n'
    sns.set(color_codes=True)
    tips = sns.load_dataset('tips')
    print (tips.head(), br)
    X = tips.drop(['tip'], axis=1).values
    y = tips['tip'].values
    print (X.shape, y.shape)