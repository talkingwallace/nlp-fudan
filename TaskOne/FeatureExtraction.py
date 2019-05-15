"""
n-gram
bag of words
"""
import pandas as pd
import numpy as np
from scipy.sparse import random
from scipy.sparse import dok_matrix # fast O(1) access
from scipy.sparse import csr_matrix

def combineDs(df):
    g = df.groupby('SentenceId')
    ids = []
    sentences = []
    labels = []
    for i in g:
        id = i[0]
        data = i[1]
        sent = ''
        for s in data['Pharse']:
            sent += s
        ids.append(id)
        sentences.append(sent)

def getVocab(df):
    rs = df['Phrase'].apply(lambda x:x.replace(',','').replace('.','')
                            .replace(';','').replace('?','').replace('!','').split(' '))
    vocab = set()
    for r in rs:
        vocab.update(r)
    return vocab,rs

def nGram(df):
    pass

def bagOfWords(df):

    vocab,seg = getVocab(df)
    vocab = list(vocab)
    d = dict(zip(vocab,[i for i in range(len(vocab))]))

    mat = random(len(df),len(vocab),0)
    mat = dok_matrix(mat)

    for (id,i) in enumerate(seg):
        print(id,'/',len(seg))
        for k in i:
            try:
                mat[id,d[k]] += 1
            except:
                pass
    return mat,vocab

def preprocess(type='bow'):
    df = pd.read_csv(r'./train.tsv', sep='\t')
    bow, vocab = bagOfWords(df)
    return df,bow,vocab

if __name__ == '__main__':

    df = pd.read_csv(r'./train.tsv',sep='\t')
    bow,vocab = bagOfWords(df)


