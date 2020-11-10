import os

path='豆瓣多轮-3-B/'
files=os.listdir(path)


for file in files:
    w=open('datas/'+file,'w')
    if file=='readme.txt':
        continue
    datas=open(path+file).readlines()
    for data in datas:
        if data.split('\t')[0]=='1':
            w.writelines('\t'.join(data.split('\t')[1:]))