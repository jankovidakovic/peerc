import re
import string
from spacy.lang.en import English
# from spacy.tokenizer import Tokenizer
from dataset import emotion2label

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def pr(train_term_doc, filter):
    p = train_term_doc[filter].sum(0)
    return (p + 1) / (filter.sum() + 1)


def get_mdl(train_term_doc, x_label_column, y_label):
    # numerator = pr(train_term_doc, (y_label == x_label_column))

    r = np.log(pr(train_term_doc, (y_label == x_label_column)) / pr(train_term_doc, (y_label != x_label_column)))
    m = LogisticRegression(C=4, dual=False)
    x_nb = train_term_doc.multiply(r)
    return m.fit(x_nb, x_label_column), r


def nbSvmDo(train: pd.DataFrame, test: pd.DataFrame):
    TEXT = 'text'
    n = train.shape[0]

    def tokenize(s):
        return [token.text for token in English().tokenizer(s)]

    min_df = 1
    max_df = 0.9
    vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                          min_df=min_df, max_df=max_df, strip_accents='unicode', use_idf=1,
                          smooth_idf=True, sublinear_tf=True)
    train_term_doc = vec.fit_transform(train[TEXT])
    test_term_doc = vec.transform(test[TEXT])

    x = train_term_doc
    test_x = test_term_doc

    labels = emotion2label.keys()

    preds = np.zeros((len(test), len(labels)))
    label_column = train['label']

    for i, label in enumerate(labels):
        print('fit', label)
        # subset = train[(train['label'] == label)]
        # num = subset.shape[0]
        m, r = get_mdl(train_term_doc, label_column.values, label)
        preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]

    print(train_term_doc)
