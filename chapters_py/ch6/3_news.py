# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV,\
     cross_val_score

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

def see_time(note):
    end = time.perf_counter()
    elapsed = end - start
    print (note,
           hf.format_timespan(elapsed, detailed=True))

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
    mnb = MultinomialNB()
    tf = TfidfVectorizer()
    print (mnb, br)
    print (tf, br)
    pipe = make_pipeline(tf, mnb)
    pipe.fit(train.data, train.target)
    labels = pipe.predict(test.data)
    f1 = f1_score(test.target, labels, average='micro')
    print ('f1_score', f1, br)
    print (pipe.get_params().keys(), br)
    param_grid = {'tfidfvectorizer__ngram_range':
                  [(1, 1), (1, 2)],
                  'tfidfvectorizer__use_idf': [True, False],
                  'multinomialnb__alpha': [1e-2, 1e-3],
                  'multinomialnb__fit_prior': [True, False]}
    start = time.perf_counter()
    rand = RandomizedSearchCV(pipe, param_grid, cv=3,
                              n_jobs = -1, random_state=0,
                              n_iter=16, verbose=2)
    rand.fit(train.data, train.target)
    see_time('RandomizedSearchCV tuning time:')
    bp = rand.best_params_
    print ()
    print ('best parameters:')
    print (bp, br)
    rbs = rand.best_score_
    mnb = MultinomialNB(alpha=0.01)
    tf = TfidfVectorizer(ngram_range=(1, 1), use_idf=False)
    pipe = make_pipeline(tf, mnb)
    pipe.fit(train.data, train.target)
    labels = pipe.predict(test.data)
    f1 = f1_score(test.target, labels, average='micro')
    print ('f1_score', f1, br)
    file = 'data/bp_news'
    np.save(file, bp)
    # need allow_pickle=True parameter
    bp = np.load('data/bp_news.npy', allow_pickle=True)
    bp = bp.tolist()
    print ('best parameters:')
    print (bp, br)
    start = time.perf_counter()
    scores = get_cross(pipe, train.data, train.target)
    see_time('cross-validation:')
    print (np.mean(scores))