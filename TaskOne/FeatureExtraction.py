"""
n-gram
bag of words
"""
import pandas as pd
import numpy as np
from scipy.sparse import random
from scipy.sparse import dok_matrix # fast O(1) access
from scipy.sparse import csr_matrix
from scipy.sparse import  csc_matrix

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

def sigmoid(arr):
    return 1/(1+np.exp(-arr))

if __name__ == '__main__':


    df = pd.read_csv(r'./train.tsv',sep='\t')
    bow,vocab = bagOfWords(df)

    """
    Linear Model
    """

    """
    Class Cate 0 1 2 | 3 4
    分为两类 做Logistic回归
    """

    para = {
        'lr':50,# 学习率
        'miniBatch':False,
        'epoch':2000,
    }

    baseline = df.groupby('Sentiment').count().max()[0]/len(df)
    classNum = len(df.groupby('Sentiment').count())
    df['biClass'] = df['Sentiment'].apply(lambda x:0 if x<=2 else 1)
    # 标准正态分布初始化
    weight = np.random.randn(len(vocab))
    # 方便计算
    bow = csc_matrix(bow)

    N = len(df) # # of samples
    target = np.array(df['biClass']) # true label

    for i in range(para['epoch']):
        print('starting epoch:',i)
        inter = bow*weight
        inter = sigmoid(inter)
        predict = np.logical_not(inter<=0.7)
        np.equal(predict,target)
        f = np.equal(predict, target)
        print('acc:',len(f.nonzero()[0])/len(f))
        # loss function -(1/N)*Sum(y(logy_hat)+(1-y)(log(1-y_hat)))
        loss = 0
        pos = np.log(inter)*target
        neg = np.log(1-inter)*(np.logical_not(target)+0)
        loss = -(1/N)*np.sum(pos+neg)

        # gradient descent
        # gradient = -(1/N)*Sum(x_n*(y_n - y_hat))
        gd = (-1/N)*(target - inter)*bow # ????
        weight = weight - para['lr']*gd
        print('loss:',loss)
        # print(weight)
        print('*'*100)

    """
    Class Cate 0 1 2 3 4
    做Softmax回归
    """
