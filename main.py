import random
from utils import *
from module.model import Model
from dataset import Model_Dataset
from dataset import Vocab
from torch.utils.data import DataLoader
import yaml
import torch

config=yaml.load(open('config.yaml').read())

lr=config['lr']
path=config['path']
test_path=config['test_path']
embedding_size=config['embedding_size']
num_layers=config['num_layers']
num_heads=config['num_heads']
dropout=config['dropout']
max_len=config['max_len']
hidden=config['hidden']
batch_size=config['batch_size']
epoches=config['epoches']
devices=[i for i in range(config['devices'])]
mul_num=config['mul_num']
pretrain_path=config['pretrain_path']
beam_num=config['beam_num']
beam_size=config['beam_size']
prob=config['prob']
vocab_size=config['vocab_size']
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datas=list(map(lambda x:x.split('\t'),open(path).readlines()))
test_datas=list(map(lambda x:x.split('\t'),open(test_path).readlines()))

vocab=Vocab(vocab_size)
def create_data(datas):
    new_datas=[]
    for i,data in enumerate(datas):
        data=list(map(lambda x:x.strip().split(' '),data))
        for i in range(len(data)-1):
            vocab.s2i_f(data[i])
            vocab.s2i_f(data[i+1])
            da=[data[i],data[i+1],data[i-1]] if i>0 else [data[i],data[i+1],None]
            new_datas.append(da)
    return new_datas
new_datas=create_data(datas)
print(len(new_datas))
new_test_datas=create_data(test_datas)
vocab.clean_s2i()

dataset=Model_Dataset(new_datas,max_len,vocab.s2i,mul_num)
test_dataset=Model_Dataset(new_test_datas,max_len,vocab.s2i,mul_num)
dataloader=DataLoader(dataset,batch_size,shuffle=True)
valid_dataloader=DataLoader(test_dataset,1,shuffle=True)

model=Model(embedding_size,num_layers,num_heads,max_len,hidden,len(vocab.s2i),0,dropout,prob).to(device)
optim=torch.optim.Adam(model.parameters(),0.0002)
step_decay=[i for i in range(100,5000,50)]
step_decay+=[i for i in range(5000,100000,500)]
step_decay=torch.optim.lr_scheduler.MultiStepLR(optim,step_decay,0.99)
loss_fn=nn.NLLLoss(ignore_index=0).to(device)
#loss_fn=LabelSmoothing(len(vocab.s2i),0,0.2)

if torch.cuda.is_available():
  model=torch.nn.DataParallel(model,device_ids=devices)

if pretrain_path:
    model.load_state_dict(torch.load(pretrain_path))

#one_index=list(range(0,batch_size*max_len,max_len))

def label_smoothing(y_pred,y_true,high,padding_index):
    size=y_pred.shape[-1]

    mask=1-torch.eq(y_true,padding_index).to(torch.float)
  #  mask[one_index]=one_weight
  #  print(mask)
    weight=torch.zeros_like(y_pred)
    weight.fill_((1-high)/(size-1))
    weight.scatter_(1,y_true.unsqueeze(-1),high)
    loss=-y_pred.mul(weight).sum(-1)
    loss=loss.mul(mask)
    return loss.sum()/mask.sum()

def select_top_k(predictions, k=5):
    if random.random()<0.5:
        predicted_index = random.choice(
            predictions[0, -1, :].sort(descending=True)[1][1:k])
    else:
        predicted_index=torch.argmax(predictions[:, -1],-1)[0]
    return predicted_index

def test_out(model,num,start_index,dataloader):
    for j in range(num):
        x,prev_x,y,label,x_mask,prev_x_mask,y_mask =dataloader.__iter__().__next__()

        tgt=torch.ones([x.shape[0],1],dtype=torch.long).to(device)*start_index
        '''
        for i in range(beam_num):
            tgt,res=beam_search(model,2,x,prev_x,x_mask,tgt,prev_x_mask,beam_size,0,0)
            tgt=tgt.unsqueeze(0)
        '''
        for i in range(max_len):
            predict=model(x.to(device),prev_x.to(device),tgt.to(device),x_mask.to(device),prev_x_mask.to(device),None)

            predict_indexes=select_top_k(predict,10).unsqueeze(-1).unsqueeze(-1)
            tgt=torch.cat([tgt,predict_indexes],-1)
            if predict_indexes==vocab.s2i['</s>']:
              break

        print('###'+vocab.i2s_f(x[-1]))
        print('###'+vocab.i2s_f(tgt[-1]))
        print()


print(len(vocab.s2i))

steps=0
for epoch in range(epoches):
  for x,prev_x,y,label,x_mask,prev_x_mask,y_mask in dataloader:

      x,prev_x,y,label,x_mask,prev_x_mask,y_mask=map(lambda x:x.to(device),[x,prev_x,y,label,x_mask,prev_x_mask,y_mask])
      predict=model(x,prev_x,y,x_mask,prev_x_mask,y_mask)


      seq = predict.argmax(-1)

      print(vocab.i2s_f(seq[-1]))
      print(vocab.i2s_f(label[-1]))
      loss=loss_fn(predict.view(-1, len(vocab.s2i)), label.view(-1))
 #     loss = label_smoothing(predict.view(-1, len(vocab.s2i)), label.view(-1), 0.8, 0)
      print(epoch,steps,loss)

      optim.zero_grad()

      loss.backward()
      optim.step()
      steps+=1
#      test_out(model,5,vocab.s2i['<s>'],valid_dataloader)

      if steps%1000==0:
          torch.save(model.state_dict(),'./model1.pt')

          test_out(model,2,vocab.s2i['<s>'],valid_dataloader)

