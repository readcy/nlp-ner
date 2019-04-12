# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:48:21 2019

@author: hasee
"""
import os
from gensim.models import word2vec
import logging
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product,count
import numpy as np

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
path='C:\\Users\\hasee\\Documents\\news\\'
stopword_path='C:\\Users\\hasee\\Documents\\stopwords.txt'
output_path='C:\\Users\\hasee\\Documents\\news_abstract\\'


        
 
#################################################################################3
wordvec=word2vec.Word2Vec.load('wiki.zh.text.model')

#cut sentence
def cut_sentence(sentence):
    puns=frozenset(u'。！？.?:!')
    tmp=[]
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp=[]
    yield ''.join(tmp)




# stop words

def creat_stopwords(path=stopword_path):
    stoplist=[]
    with open(path,'r',encoding='utf-8') as fp:
        for word in fp.readlines():
            stoplist.append(word.strip())
    return stoplist

#two sentence similarity use token count
def similarity_naive(sentence1,sentence2):
    counter=0
    for word in sentence1:
        for word in sentence2:
            counter+=1
    return counter/(math.log(len(sentence1))+math.log(len(sentence2)))


#two sentence similarity use cosine similarity
def similarity_cosine(sentence1,sentence2,ep=1e-9):
    sent1_array=np.array(sentence1)
    sent2_array=np.array(sentence2)
    cosine_1=np.sum(sent1_array*sent2_array)
    cosine_21=np.sqrt(sum(sent1_array**2))
    cosine_22=np.sqrt(sum(sent2_array**2))
    cosine_value=cosine_1/float(cosine_21+cosine_22+ep)
    return cosine_value

#creat sentence relevent graph

def creat_graph(sentence_list,wordvec,similarity):
    num=len(sentence_list)
    graph=np.zeros((num,num))
    for i,j in product(range(num),repeat=2):
        if i !=j:
            graph[i,j]=comput_similarity_by_avg(sentence_list[i],sentence_list[j],wordvec,similarity)
    return graph

#comput similarity use word vector

def comput_similarity_by_avg(sentence1,sentence2,wordvec,similarity):
    if len(sentence1)==0 or len(sentence2)==0:
        return 0.0
    count1=0
    count2=0
    vec1=0
    vec2=0
    for word in sentence1:
        if word in wordvec:
            vec1+=wordvec[word]
            count1+=1
    for word in sentence2:
        if word in wordvec:
            vec2+=wordvec[word]
            count2+=1
    if  count1==0 or count2==0:
        return 0
    else:    
        similarity_value=similarity(vec1/count1,vec2/count2)
    
        return similarity_value
#calculate graph score using similarity_value as weight
def calculate_score(graph,scores,i,d=0.85):
    length =len(graph)
    added_score=0.0
    for j in range(length):
        fraction=0.0
        denominator=0.0
        fraction=graph[j,i]*scores[j]
        for k in range(length):
            denominator+=graph[j,k]
            if denominator==0:
                denominator=1
        added_score+=fraction/denominator
    weight_score=(1-d)+d*added_score
    return weight_score
#iteration for solving sorce
def iteration(graph,different):
    scores=[0.5 for _ in range(len(graph))]
    old_scores=[0.0 for _ in range(len(graph))]
    while different(scores,old_scores):
        for i in range(len(graph)):
            old_scores[i]=scores[i]
        for j in range(len(graph)):
            scores[j]=calculate_score(graph,scores,j)
    return scores
#iteration condition
def different(scores,old_scores):
    flag=False
    for i in range(len(scores)):
        if math.fabs(scores[i]-old_scores[i])>=0.01:
            flag=True
            break
    return flag

#utils for fillter word
def filter_symbols(sents):
    stopwords=creat_stopwords()+['。','','.']
    _sents=[]
    for sentence in sents:
        for word in sentence:
            if word in stopwords:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
            
    return _sents

def filter_vec(sents,wordvec):
    _sents=[]
    for sentence in sents:
        for word in sentence:
            if word not in wordvec:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents

#jieba cut
def summarize(text,n,wordvec,similarity,different):
    tokens=cut_sentence(text)
    sentences=[]
    sents=[]
    for sent in tokens:
        sentences.append(sent)
        sents.append([word for word in jieba.cut(sent) if word])
    sents_1=filter_symbols(sents)
    sents_2=filter_vec(sents_1,wordvec)
    graph=creat_graph(sents_2,wordvec,similarity)
    scores=iteration(graph,different)
    sent_selected=nlargest(n,zip(scores,count()))
    sent_index=[]
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sentences[i] for i in sent_index]







if os.path.isdir(path):
    print('文件已经存在')
else:
    os.mkdir(path)
if os.path.isdir(output_path):
    print('输出文件存在')
else:
    os.mkdir(output_path)
for i in os.listdir(path):
    with open(path+str(i),'r') as fp:
        text=''.join(fp.readlines())
        text=text.replace('\n','')
        results=summarize(text,5,wordvec,similarity_cosine,different)
    
    with open(output_path+str(i),'w',encoding='utf-8') as f:
        for lin in results:
            f.write(''.join(lin)+'\n')
            
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

             
             
             
             
             
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





        
        

