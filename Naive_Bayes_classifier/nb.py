#!/usr/bin/env python
# coding: utf-8

# In[6]:


import re
import math
import pandas as pd
import numpy as np
from itertools import groupby
    
#loading in the training file as dataframe
df = pd.read_csv('trainingset.txt')

#applying headers to the dataframe
df.columns = ['id','age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','outcome']

#making another dataframe that only contains the lines that have the outcome value of TypeA
out = df[df["outcome"].str.contains("TypeA")]

#making another dataframe for lines that contain Type B
out2 = df[df["outcome"].str.contains("TypeB")]

#Turning both dataframes to lists
data1 = out.values.tolist()
data2 = out2.values.tolist()


# In[7]:


def getwords(doc):
    
    splitter = re.compile('\W+')

    words = [s.lower() for s in splitter.split(doc) if len(s) > 0 and len(s) < 20]
    
    return dict([(w,1) for w in words])


# In[8]:


class classifier:
    
    def __init__(self, getfeatures, filename=None):
    
        self.fc = {}
    
        self.cc = {}
        self.getfeatures = getfeatures
        
    
    def incf(self, f, cat):
    
        self.fc.setdefault(f,{})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat]+=1
        
        
    def incc(self, cat):
        
        self.cc.setdefault(cat,0)
        self.cc[cat]+=1
        
        
    def fcount(self, f, cat):
        
        if f in self.fc and cat in self.fc[f]:
            
            return float(self.fc[f][cat])
        
        
        return 0.0
    
    
    def catcount(self,cat):
        
        if cat in self.cc:
        
           return float(self.cc[cat])


        return 0
    
    
    def totalcount(self):
        
        return sum(self.cc.values())
    
    
    
    def categories(self):
        
        return self.cc.keys()
    
    
    def train(self, item, cat):
        
        features = self.getfeatures(item)
        
        for f in features:
            
            self.incf(f, cat)
            
        self.incc(cat)
    


# In[9]:


class naiveb(classifier):
    
    def __init__(self, getfeatures):
        
        classifier.__init__(self, getfeatures)
        
        self.thresholds = {}
        
    def docprob(self, item, cat):
        
        features = self.getfeatures(item)
        
        p = 1
        
        for f in features: p *= self.wprob(f, cat, self.fprob)
            
        return p
    
    
    def wprob(self, f, cat, prf, weight = 1.0, ap = 0.1):
        
        bprob = prf(f, cat)
        
        totals = sum([self.fcount(f, c) for c in self.categories()])
        
        bp = ((weight*ap) + (totals*bprob))/(weight + totals)
        
        return bp
    

    def fprob(self, f, cat):
        
        if self.catcount(cat) == 0 : return 0
        
        return self.fcount(f, cat)/self.catcount(cat)

    
    def prob(self, item, cat):
        
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        
        return docprob * catprob
    
    def setT(self, cat, t):
        
        self.threshold[cat] = t
        
        
        
    def getT(self, cat):
        
        if cat not in self.thresholds: return 1.0
        
        return self.threshold[cat]
    
    def classify(self, item, default=None):
        
        probs = {}
        max = 0.0
        
        for cat in self.categories():
            
            probs[cat] = self.prob(item, cat)
            
            if probs[cat] > max:
                
                max = probs[cat]
                best = cat
                
        
        for cat in probs:
            
            if cat == best: continue
            if probs[cat] * self.getT(best) > probs[best]: return default
            
        return best


# In[10]:


cl = naiveb(getwords)


#for loop to train the classifier for each line off the type a list
for x in data1:
    #turning each line in the list into string so could be used to train the model
    g=str(x)
    cl.train(g, 'TypeA')
    
#for loop to train the classifier for each line off the type b list  
for i in data2:
    #turning each line into string
    t=str(i)
    cl.train(t, 'TypeB')
    
#dataframe that contains queries.txt which we will use to test our model   
test = pd.read_csv('queries.txt',header=None)

#test dataframe being turned into list
tdata = test.values.tolist()

#variable to keep track of what test is being predicted
count = 0;

# for loop to loop through tdata list so that the classifier could predict each line
for j in tdata:
    #Turning test lines into strings so that it could be passed through classifier
    k = str(j)
    
    #increases count var by 1 to keep track track of test lines
    count = count + 1
    
    #outputs if the outcome for the prediction will be type a or type b
    ans = cl.classify(k , default='unknown')
    
    #making a output text file to keep track of all printed outputs, a appends it to text file
    f = open("output.txt", "a")
    print("TEST",count,ans,file=f)
    
    
f.close()


# In[ ]:




