import math
import numpy as np
import torch
import copy
class Vocab(object):
    def __init__(self,max_size=50000):
        self.s2i={'_PAD':0,'_UNK':1,'<s>':2,'</s>':3}
        self.i2s=['_PAD','_UNK','<s>','</s>']
        self.s2i_num = {'_PAD': float('inf'), '_UNK': float('inf')-1, '<s>': float('inf')-2, '</s>': float('inf')-3}
        self.p_c=None
        self.size=max_size

    #    self.special_label=['_PAD','_UNK','<s>']
        self.c_num=[]
        self.if_p_c=False
    def s2i_f(self,words):
        for word in words:
            self.s2i_num[word]=self.s2i_num.get(word,0)+1

    def clean_s2i(self):
        items=sorted(self.s2i_num,key=lambda x:self.s2i_num[x])[::-1][:self.size]

        for item in items:
            if item not in self.s2i:
                self.s2i[item]=len(self.i2s)
                self.i2s.append(item)

    def i2s_f(self,seq):
        res=''
        if len(seq.shape)==0:
            seq=[seq]
        for index in seq:
            res+=self.i2s[int(index)]
        return res

    def s2i2i2s(self):
        self.i2s=[]
        for k in self.s2i:
            self.i2s.append(k)

    def get_p(self):
        self.p_c=torch.ones([len(self.s2i,)]).to(self.device)*(math.e-1)
        self.c_num=torch.zeros([len(self.s2i,)]).to(self.device)
'''
        if not self.if_p_c:
            sum_=sum(self.p_c)
            for i in range(3,len(self.p_c)):
                tmp=self.p_c[i]/sum_
                tmp=1/math.log(1.02+tmp,math.exp(1))
                self.p_c[i]=max(min(tmp,50.0),1.0)
        self.if_p_c = True
'''