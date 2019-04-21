# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:26:27 2019

@author: admin
"""
import math
import operator
import numpy as np
import sys
from csv import DictReader
from math import exp, copysign, log, sqrt
from datetime import datetime
import random
class SimpleLogisticRegression:
    def __init__(self,alpha,feature_num):
        self.__alpha=alpha
        self.__features_num=feature_num
        self.__coef=[0.]*self.__features_num
        self.__bais=0.
        pass
    def fit (self,X,y,verbose=False):
        last_target=None
        last_step=0
        step=0
        while True:
            step+=1
            gradient=[0.]*(self.__features_num+1)
            for tx,ty in zip(X,y):
                delta=ty-self.__sigmoid(tx)
                for i ,xi in enumerate(tx):
                    gradient[i]+=delta*xi
                gradient[-1]+=delta
            gradient=list(map(lambda g:g/len(X),gradient))
            self.__coef=list(map(lambda c,g:c+self.__alpha*g,self.__coef,gradient[:-1]))
            self.__bais+=self.__alpha*gradient[-1]
            target=sum(map(lambda py,ty:ty*math.log(py)+(1-ty)*math.log(1-py),map(self.__sigmoid,X),y))/len(X)
            if last_target is None or target-last_target> 1e-8:
                last_target=target
                last_step=step
            elif step -last_step>=3:
                break
        target=sum(map(lambda py,ty:ty*math.log(py)+(1-ty)*math.log(1-py),map(self.__sigmoid,X),y))/len(X)
        return target
        
    def predict(self,X):
        if not self._check_columns(X):
            sys.stderr.write("The data to be evaluated can't match training data's features\n") 
            return None
        return map(self.__sigmoid,X)
    
        
    def __sigmoid(self,x):
        return 1./(1+math.exp(-sum(map(operator.mul,self.__coef,x))-self.__bais))
    def _check_columns(self,X):
        for x in X:
            if not isinstance(x,(list,tuple)):
                print('Input error')
                return False
            if len(x)!=self.__features_num:
                print('size error')
                return False
        return True    
            


class FM_FTRL:
    def __init__(self, fm_dim, fm_initDev, L1, L2, L1_fm, L2_fm, D, alpha, beta, alpha_fm = .1, beta_fm = 1.0, dropoutRate = 1):
        ''' initialize the factorization machine.'''
        
        self.__alpha = alpha              # learning rate parameter alpha
        self.__beta = beta                # learning rate parameter beta
        self.__L1 = L1                    # L1 regularizer for first order terms
        self.__L2 = L2                    # L2 regularizer for first order terms
        self.__alpha_fm = alpha_fm        # learning rate parameter alpha for factorization machine
        self.__beta_fm = beta_fm          # learning rate parameter beta for factorization machine
        self.__L1_fm = L1_fm              # L1 regularizer for factorization machine weights. Only use L1 after one epoch of training, because small initializations are needed for gradient.
        self.__L2_fm = L2_fm              # L2 regularizer for factorization machine weights.
        self.__fm_dim = fm_dim            # dimension of factorization.
        self.__fm_initDev = fm_initDev    # standard deviation for random intitialization of factorization weights.
        self.__dropoutRate = dropoutRate  # dropout rate (which is actually the inclusion rate), i.e. dropoutRate = .8 indicates a probability of .2 of dropping out a feature.
                                        #1 for not use dropout
        self.__D = D                      #keep the dim large to  avoid overflow
        
        # model
        # n: like adptive learning_rate
        # z: like gradiwnt
        # w: weights
        
        # let index 0 be bias term 
        self.__n = [0.] * (D + 1) 
        self.__z = [0.] * (D + 1)
        self.__w = [0.] * (D + 1)
        
        self.__n_fm = {}
        self.__z_fm = {}
        self.__w_fm = {}
    
        
    def __init_fm(self, i):
       #init the fm parameters,especially, use the gaussion init of graident z
        if i not in self.n_fm:
            self.__n_fm[i] = [0.] * self.__fm_dim
            self.__w_fm[i] = [0.] * self.__fm_dim
            self.__z_fm[i] = [0.] * self.__fm_dim
            
            for k in range(self.__fm_dim): 
                self.__z_fm[i][k] = random.gauss(0., self.__fm_initDev)
    
    def predict_raw(self, x):
       # predict the use the sum of nozero weight
        alpha = self.__alpha
        beta = self.__beta
        L1 = self.__L1
        L2 = self.__L2
        alpha_fm = self.__alpha_fm
        beta_fm = self.__beta_fm
        L1_fm = self.__L1_fm
        L2_fm = self.__L2_fm
        
        # first order weights model
        n = self.__n
        z = self.__z
        w = self.__w
        
        # FM interaction model
        #n_fm = self.__n_fm
        #z_fm = self.__z_fm
        #w_fm = self.__w_fm
        
        raw_y = 0.
        
        # calculate the bias contribution
        for i in [0]:
            # no regularization for bias
            w[i] = (- z[i]) / ((beta + sqrt(n[i])) / alpha)
            
            raw_y += w[i]
            self.w[i]=w[i]
        
        # calculate the first order contribution.
        for i in x:
            sign = -1. if z[i] < 0. else 1. # get sign of z[i]
            
            if sign * z[i] <= L1:
                w[i] = 0.
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            
            raw_y += w[i]
            self.w[i]=w[i]
        
        len_x = len(x)
        # calculate factorization machine contribution.
        for i in x:
            self.__init_fm(i)
            for k in range(self.__fm_dim):
                sign = -1. if self.__z_fm[i][k] < 0. else 1.   # get the sign of z_fm[i][k]
                
                if sign * self.__z_fm[i][k] <= L1_fm:
                    self.__w_fm[i][k] = 0.
                else:
                    self.__w_fm[i][k] = (sign * L1_fm - self.__z_fm[i][k]) / ((beta_fm + sqrt(self.__n_fm[i][k])) / alpha_fm + L2_fm)
        
        for i in range(len_x):
            for j in range(i + 1, len_x):
                for k in range(self.__fm_dim):
                    raw_y += self.__w_fm[x[i]][k] * self.__w_fm[x[j]][k]
        
        return raw_y
    
    def predict(self, x):
        ''' predict the logit
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x), 35.), -35.)))
    
    def __dropout(self, x):
        ''' dropout variables in list x
        '''
        for i, var in enumerate(x):
            if random.random() > self.__dropoutRate:
                del x[i]
    
    def dropoutThenPredict(self, x):
        ''' first dropout some variables and then predict the logit using the dropped out data.
        '''
        self.__dropout(x)
        return self.predict(x)
    
    def predictWithDroppedOutModel(self, x):
        ''' predict using all data, using a model trained with dropout.
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x) * self.__dropoutRate, 35.), -35.)))
    
    def update(self, x, p, y):
        ''' Update the parameters using FTRL (Follow the Regularized Leader)
        '''
        alpha = self.__alpha
        alpha_fm = self.__alpha_fm
        
        # model
        n = self.__n
       # z = self.__z
        w = self.__w
        
        # FM model
        n_fm = self.__n_fm
       # z_fm = self.__z_fm
        w_fm = self.__w_fm
        
        # cost gradient with respect to raw prediction.
        g = p - y
        
        fm_sum = {}      # sums for calculating gradients for FM.
        len_x = len(x)
        
        for i in x + [0]:
            # update the first order weights.
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            self.__z[i] += g - sigma * w[i]
            self.__n[i] += g * g
            
            # initialize the sum of the FM interaction weights.
            fm_sum[i] = [0.] * self.__fm_dim
        
        # sum the gradients for FM interaction weights.
        for i in range(len_x):
            for j in range(len_x):
                if i != j:
                    for k in range(self.__fm_dim):
                        fm_sum[x[i]][k] += w_fm[x[j]][k]
        
        for i in x:
            for k in range(self.fm_dim):
                g_fm = g * fm_sum[i][k]
                sigma = (sqrt(n_fm[i][k] + g_fm * g_fm) - sqrt(n_fm[i][k])) / alpha_fm
                self.__z_fm[i][k] += g_fm - sigma * w_fm[i][k]
                self.__n_fm[i][k] += g_fm * g_fm
    
    def write_w(self, filePath):
        ''' write out the first order weights w to a file.
        '''
        with open(filePath, "w") as f_out:
            for i, w in enumerate(self.w):
                f_out.write("%i,%f\n" % (i, w))
    
    def write_w_fm(self, filePath):
        ''' write out the factorization machine weights to a file.
        '''
        with open(filePath, "w") as f_out:
            for k, w_fm in self.w_fm.iteritems():
                f_out.write("%i,%s\n" % (k, ",".join([str(w) for w in w_fm])))


def logLoss(p, y):
    ''' 
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    p = max(min(p, 1. - 1e-15), 1e-15)
    return - log(p) if y == 1. else -log(1. - p)



#the hash map calculate the index of input,which value is not 0
def data(filePath, hashSize, hashSalt):
    ''' generator for data using hash trick
    
    INPUT:
        filePath
        hashSize
        hashSalt: String with which to salt the hash function
    '''
    
    for t, row in enumerate(DictReader(open(filePath))):
        ID = row['id']
        del row['id']
        
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']
        
        date = int(row['hour'][4:6])
        
        row['hour'] = row['hour'][6:]
        
        x = []
        
        for key in row:
            value = row[key]
            
            index = abs(hash(hashSalt + key + '_' + value)) % hashSize + 1      # 1 is added to hash index because I want 0 to indicate the bias term.
            x.append(index)
        
        yield t, date, ID, x, y
            
            
            
            
            
        
        
        
        


































        
            
            
            
            
        
                
