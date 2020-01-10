from sklearn.datasets import fetch_20newsgroups

if __name__ == "__main__":
    br = '\n'
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    print ('data:')
    print (train.target.shape, 'shape of train data')
    print (test.target.shape, 'shape of test data', br)
    targets = test.target_names
    print (targets, br)
    categories = ['rec.autos', 'rec.motorcycles', 'sci.space',
                  'sci.med']
    train = fetch_20newsgroups(subset='train',
                               categories=categories)
    test = fetch_20newsgroups(subset='test',
                              categories=categories)
    print ('data subset:')    
    print (train.target.shape, 'shape of train data')
    print (test.target.shape, 'shape of test data', br)
    targets = train.target_names
    print (targets)