import torch

class Node:
    def __init__(self,name):
        self.value=None
        self.name=name
        self.child=[]


def beam_search(model,topn,x,x_mask,tgt,max_len,i,res):
    node=None
    if i==0:
        node=Node(str(i))

    if i==max_len:
        return [tgt[-1]],res

    predict=model(x,tgt,x_mask,None)[0,-1]
    predict_indexs=torch.argsort(predict)[-topn:]
    res_num=0
    res_seq=[]
    for predict_index in predict_indexs:
        res_new=predict[predict_index]
        node.value=res
        new_tgt=torch.cat([tgt,predict_index.unsqueeze(0)],-1)
        seq,new_res_num=beam_search(model,topn,x,x_mask,new_tgt,start,max_len,i+1,res_new)
        if res_num<new_res_num+res:
            res_num=new_res_num+res
            res_seq=[tgt[-1]]+seq
    return res_seq,res_num





