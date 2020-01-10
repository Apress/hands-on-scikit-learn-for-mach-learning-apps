import pandas as pd

if __name__ == "__main__":
    br = '\n'
    f = 'data/bank.csv'
    bank = pd.read_csv(f)
    features = list(bank)
    print (features, br)
    X = bank.drop(['y'], axis=1).values
    y = bank['y'].values
    print (X.shape, y.shape, br)
    print (bank[['job', 'education', 'age', 'housing',
                 'marital', 'duration']].head())