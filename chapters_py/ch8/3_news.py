import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict_category(s, m, t):
    pred = m.predict([s])
    return t[pred[0]]

if __name__ == "__main__":
    br = '\n'
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    categories = ['rec.autos', 'rec.motorcycles',
                  'sci.space', 'sci.med']
    train = fetch_20newsgroups(subset='train',
                               categories=categories,
                               remove=('headers', 'footers',
                                       'quotes'))
    test = fetch_20newsgroups(subset='test',
                              categories=categories,
                              remove=('headers', 'footers',
                                      'quotes'))
    targets = train.target_names
    print ('targets:')
    print (targets, br)
    # need allow_pickle=True parameter
    bp = np.load('data/bp_news.npy', allow_pickle=True)
    bp = bp.tolist()
    print ('best parameters:')
    print (bp, br)
    mnb = MultinomialNB(alpha=0.01, fit_prior=False)
    tf = TfidfVectorizer(ngram_range=(1, 1), use_idf=False)
    pipe = make_pipeline(tf, mnb)
    pipe.fit(train.data, train.target)
    labels = pipe.predict(test.data)
    f1 = f1_score(test.target, labels, average='micro')
    print ('f1_score', f1, br)
    labels = pipe.predict(test.data)
    cm = confusion_matrix(test.target, labels)
    plt.figure('confusion matrix')
    sns.heatmap(cm.T, square=True, annot=True, fmt='d',
                cmap='gist_ncar_r',
                xticklabels=train.target_names,
                yticklabels=train.target_names, cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.tight_layout()
    print ('***PREDICTIONS***:')
    doc1 = 'imagine the stars ...'
    doc2 = 'crashed on highway without seatbelt'
    doc3 = 'dad hated the medicine ...'
    y_pred = predict_category(doc1, pipe, targets)
    print (y_pred)
    y_pred = predict_category(doc2, pipe, targets)
    print (y_pred)
    y_pred = predict_category(doc3, pipe, targets)
    print (y_pred)
    plt.show()