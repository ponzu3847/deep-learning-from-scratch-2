# -*- coding: utf-8 -*-
"""functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ye8-efS5F3Gjxpyggd9BMbhXjun3rctE
"""

from common.np import *

def softmax(x):
    if x.ndim==2:
        x=x-x.max(axis=1,keepdims=True) #オーバーフロー対策
        x=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    elif x.ndim==1:
        x=x-x.max() #オーバーフロー対策
        x=np.exp(x)/np.sum(np.exp(x))

    return x

def cross_entropy_error(y,t):
    if y.ndim==1:   #バッチサイズが１の場合
        y=np.reshape(1,y.size)
        t=np.reshape(1,t.size)

    if t.ndim!=1:   #tがone_hot-vectorの場合、ラベル表記に変換する
        t=np.argmax(t,axis=1)
    batch_size=y.shape[0]
    
    #y[np.arange(batch_size),t]はnumpyのファンシーインデックスを使っている
    loss=-np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

    return loss

def sigmoid(x):
    return 1/(1+np.exp(-x))