# -*- coding: utf-8 -*-
"""layers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aJQMx1zYPDBiYX4mH9vCZYLvlAtA5iba
"""

from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import *

class MatMul:
    def __init__(self,W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.x=None

    def forward(self,x):
        W,=self.params
        out=np.dot(x,W)
        self.x=x
        return out

    def backward(self,dout):
        W,=self.params
        dx=np.dot(dout,W.T)
        dW=np.dot(self.x.T,dout)
        self.grads[0][...]=dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params=[]
        self.grads=[]
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self,dout):
        dx=self.out*(1.-self.out)*dout
        return dx

class Affine:
    def __init__(self,W,b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W),np.zeros_like(b)]
        self.x=None

    def forward(self,x):
        W,b=self.params
        out=np.dot(x,W)+b
        self.x=x
        return out

    def backward(self,dout):
        W,b=self.params
        db=np.sum(dout,axis=0)
        dx=np.dot(dout,W.T)
        dW=np.dot(self.x.T,dout)
        self.grads[0][...]=dW
        self.grads[1][...]=db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params=[]
        self.grads=[]
        self.y=None
        self.t=None

    def forward(self,x,t):
        self.y=softmax(x)
        self.t=t
        if self.t.size==self.y.size:    #tがone-hot-vectorの場合ラベル表記に変換する
            self.t=np.argmax(self.t,axis=1)
        # if self.t.ndim!=1:
            # self.t=np.argmax(self.t,axis=1)
        loss=cross_entropy_error(self.y,self.t)
        return loss

    def backward(self,dout=1):
        #backward時のself.tはforward時にラベル表記に変換されている
        batch_size=self.t.shape[0]
        dx=self.y.copy()
        dx[np.arange(batch_size),self.t]-=1 #one_hot_vectorの１を引く
        dx*=dout
        dx=dx/batch_size
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params,self.grads=[],[]
        self.loss=None
        self.y=None
        self.t=None

    def forward(self,x,t):
        self.y=1/(1+np.exp(-x))
        self.t=t

        #t=0(負例)の場合はnp.c_[1-self.y,self.y][0]、すなわち1-self.yの確率を使用して交差エントロピー誤差を計算する
        #t=0 or 1のため、tの値がそのままnp.c_[1-self.y,self.y]のインデックスになる
        self.loss=cross_entropy_error(np.c_[1-self.y,self.y],self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size*dout

        return dx

class Embedding:
    def __init__(self,W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.idx=None

    def forward(self,idx):
        W,=self.params
        self.idx=idx
        out=W[idx]
        return out

    def backward(self,dout):
        dW,=self.grads
        dW[...]=0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            # dW[self.idx]=dout #←重複したidxに対応できない誤った実装
            # for i,word_id in enumerate(self.idx): #正しいが処理が遅い実装
            #     dout[word_id]+=dout[i]

            #重複したidxの要素は加算する
            np.add.at(dW, self.idx, dout)
        return None

class Softmax:
    def __init__(self):
        self.params,self.grads=[],[]
        self.out=None

    def forward(self,x):
        self.out=softmax(x)
        return self.out

    def backward(self,dout):
        dx=dout*self.out
        sumdx=np.sum(dx,axis=1,keepdims=True)
        dx -= self.out * sumdx
        return dx