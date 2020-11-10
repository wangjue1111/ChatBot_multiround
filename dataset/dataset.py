
from torch.utils.data import Dataset
import numpy as np
import torch

def create_seq_mask(item,vocab1,new_item=None,vocab2=None):
    mask=np.equal(item,vocab1['_PAD'])
    if new_item!=None:
        new_mask=np.not_equal(new_item,vocab2['_PAD'])
        mask=mask|new_mask
    try:
        return mask.astype(np.float32)
    except:
        return mask.to(torch.float32)

def create_attn_mask(size):
    mark = 1-torch.triu(torch.ones(size,size)).T
    return mark.numpy()

def to_array(items):
    return list(map(lambda x:np.array(x,np.long),items))

def create_seq(items,vocab,max_len,flag=0):

    res=[]

    if flag==1:
        items=['<s>']+items

    elif flag==2:
        items=items+['</s>']

    for item in items:

        res.append(vocab.get(item,vocab['_UNK']))
    length=len(res)
    if len(res)<max_len:
        res+=[vocab['_PAD'] for i in range(max_len-len(res))]
    elif len(res)>max_len:
        res=res[:max_len]
    return res

def create_label(labels,max_len,vocab,length):
    labels=labels.split('\n')
    label2seq=torch.zeros([max_len,])
    label2seq[:length]=1
    for label in labels:
        label=label.split('\t')[1].split(' ')
        ann=vocab[label[0]]
        label2seq[int(label[1]):int(label[2])+1]=ann
    return label2seq

def create_pad_seq(items,num,max_len):
    if len(items)<num:
        items+=[[0]*max_len for i in range(num-len(items))]
    else:
        items=items[:num]
    return items


class Model_Dataset(Dataset):
    def __init__(self,datas,max_len,vocab,num_mul):
        self.datas=datas
        self.vocab=vocab
        self.max_len=max_len
        self.mul_num=num_mul

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data=self.datas[item][0]
        prev_data=self.datas[item][2]
        data=create_seq(data,self.vocab,self.max_len)
        prev_data=create_seq(prev_data,self.vocab,self.max_len) if prev_data is not None else [0]*self.max_len
        tgt=create_seq(self.datas[item][1],self.vocab,self.max_len,1)
        label=create_seq(self.datas[item][1],self.vocab,self.max_len,2)
        x_mask=create_seq_mask(data,self.vocab)[np.newaxis,np.newaxis,:]
        y_mask=create_seq_mask(tgt,self.vocab)[np.newaxis,np.newaxis,:]+create_attn_mask(self.max_len)[np.newaxis,:]
        prev_x_mask=create_seq_mask(prev_data,self.vocab)[np.newaxis,np.newaxis,:]
        return np.array(data),np.array(prev_data),np.array(tgt),np.array(label),x_mask,prev_x_mask,y_mask





