# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:24:57 2019

@author: admin
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold


df = pd.read_csv("ctr_data.csv")
cols = ['id','C1','banner_pos', 'site_domain',  'site_id', 'site_category','app_id', 'app_category',  'device_type',  'device_conn_type', 'C14', 'C15','C16']
lbl = preprocessing.LabelEncoder()
col=df.columns
X=df.drop(['click','hour'],axis=1)
label=df[col[1]]


count=X['C20'].value_counts()
dit=dict(count)
look_tabel=[]
for key,num in dit.items():
    if key!=-1:
        look_tabel.extend([key]*num)        
#sample 
sample_data=np.random.choice(look_tabel,len(X)-len(look_tabel)) 
X['C20']=X['C20'].replace([-1]*len(sample_data),sample_data)
categroy=('id','C1','banner_pos','site_domain','site_id','site_category','app_id','app_category','app_domain','device_id','device_ip','device_model','device_type','device_conn_type','C18','C20')   
X['site_domain'] = lbl.fit_transform(X['site_domain'].astype(str))#将提示的包含错误数据类型这一列进行转换
X['site_id'] = lbl.fit_transform(X['site_id'].astype(str))
X['site_category'] = lbl.fit_transform(X['site_category'].astype(str))
X['app_id'] = lbl.fit_transform(X['app_id'].astype(str))
X['app_category'] = lbl.fit_transform(X['app_category'].astype(str))
X['app_domain']=lbl.fit_transform(X['app_domain'].astype(str))
X['device_id']=lbl.fit_transform(X['device_id'].astype(str))
X['device_ip']=lbl.fit_transform(X['device_id'].astype(str))
X['device_model']=lbl.fit_transform(X['device_model'].astype(str))
X['device_type']=lbl.fit_transform(X['device_type'].astype(str))
X['device_conn_type']=lbl.fit_transform(X['device_conn_type'].astype(str))
X['id']=lbl.fit_transform(X['id'].astype(str))
X['C1']=lbl.fit_transform(X['C1'].astype(str))
X['banner_pos']=lbl.fit_transform(X['banner_pos'].astype(str))
X['C18']=lbl.fit_transform(X['C18'].astype(str))
X['C20']=lbl.fit_transform(X['C20'].astype(str))
for name in categroy:
    
    dum1=pd.get_dummies(X[name],prefix=name)
    X=X.join(dum1)
    X=X.drop([name],axis=1)
X['C14']=np.log(X['C14'].values+1)
X=X.join(pd.get_dummies(pd.cut(X['C14'],5,labels=False),prefix='C14'))
X=X.drop(['C14'],axis=1)
X['C15']=np.log(X['C15'].values+1)
X['C16']=np.log(X['C16'].values+1)
X['C17']=np.log(X['C17'].values+1)
X['C19']=np.log(X['C19'].values+1)
X=X.join(pd.get_dummies(pd.cut(X['C15'].values,3,labels=False),prefix='C15'))
X=X.drop(['C15'],axis=1)
X=X.join(pd.get_dummies(pd.cut(X['C16'].values,3,labels=False),prefix='C16'))
X=X.drop(['C16'],axis=1)
X=X.join(pd.get_dummies(pd.cut(X['C17'].values,3,labels=False),prefix='C17'))
X=X.drop(['C17'],axis=1)
X=X.join(pd.get_dummies(pd.cut(X['C19'].values,3,labels=False),prefix='C19'))
X=X.drop(['C19'],axis=1)
X=X.join(pd.get_dummies(pd.cut(X['C21'].values,3,labels=False),prefix='C21'))
X=X.drop(['C21'],axis=1)
label=label.values
label_train=label[0:-1000]
label_test=label[-1000:]
X=X.values
X_train=X[0:-1000,:]
X_test=X[-1000:,:]

clfs = [LogisticRegression(penalty='l2',C=0.1,max_iter=100),
        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]
n_folds = 5
skf = list(StratifiedKFold(label_train, n_folds))

# 创建零矩阵
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    print(j, clf)
    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    for i, (train, eva) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        print("Fold", i)
        x_tr,y_tr,x_eval,y_eval = X_train[train], label_train[train], X_train[eva], label_train[eva]
        clf.fit(x_tr, y_tr)
        y_submission = clf.predict_proba(x_eval)[:, 1]
        dataset_blend_train[eva, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
 
# 用建立第二层模型
clf2 = LogisticRegression(penalty='l2',C=0.1,max_iter=100)
clf2.fit(dataset_blend_train, label_train)
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]

from sklearn import metrics

fpr,tpr,thresholds = metrics.roc_curve(label_test,y_submission,pos_label=1)

auc=metrics.auc(fpr,tpr)
print(auc)
lr=LogisticRegression(penalty='l2',C=1)
lr.fit(X_train,label_train)
y_pred=lr.predict_proba(X_test)
fp,tp,thresholds=metrics.roc_curve(label_test,y_pred[:,1],pos_label=1)
auc1=metrics.auc(fp,tp)
print(auc1)













     
        
        
        