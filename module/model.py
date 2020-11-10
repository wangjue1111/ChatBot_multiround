import torch
import torch.nn as nn
from module.encoder import Encoder
from module.decoder import Decoder


class Model(nn.Module):
    def __init__(self,embedding_size,num_layers,num_heads,max_len,hidden,token_vocab_size,padding_index,dropout,prob):
        super(Model, self).__init__()
        self.encoder=Encoder(embedding_size,num_layers,num_heads,max_len,hidden,token_vocab_size,padding_index,dropout)
        self.decoder=Decoder(embedding_size,num_layers,num_heads,max_len,hidden,token_vocab_size,padding_index,dropout,prob)
        self.max_len=max_len
        self.dropout=dropout
        self.output=nn.Sequential(nn.Linear(embedding_size,token_vocab_size),nn.LogSoftmax(-1))
        self.softmax=nn.Sequential(nn.Linear(embedding_size,token_vocab_size),nn.Softmax(-1))
    def forward(self,x,prev_x,tgt,x_mask,prev_x_mask,tgt_mask):
        encoder_outputs=self.encoder(x,x_mask,self.dropout)
        prev_encoder_outputs=self.encoder(prev_x,prev_x_mask,self.dropout) if prev_x is not None else None
        mems=torch.cat([encoder_outputs,prev_encoder_outputs],1) if prev_encoder_outputs is not None else encoder_outputs
        src_mask=torch.cat([x_mask,prev_x_mask],-1) if prev_encoder_outputs is not None else x_mask
        outputs=self.decoder(tgt,mems,tgt_mask,src_mask,self.dropout)
        a=self.output(outputs)
     #   b=self.softmax(outputs)

        return a

