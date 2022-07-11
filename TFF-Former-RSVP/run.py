from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
import glob
import time
import math
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import collections
import random
from tqdm import tqdm
from torch import nn
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import config
from model import Transformer
from metric import acc10, GELU, Corr, att_norm, LabelSmoothingLoss, LabelSmoothingLoss1, KLloss, accuracy

#GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'


dist.init_process_group(backend="nccl")#, init_method="env://", world_size=torch.cuda.device_count(),rank=local_rank)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']

path1 = os.path.abspath(os.curdir)+'/data/'
path2 = '/Task1.npz'

epochs = config.epoch 

BNmatrix = np.zeros(len(name))
PRmatrix = np.zeros((len(name),2))

for id_name in range(len(name)):
    name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']
    name_test = [name[id_name]]
    name_train = name
    del name_train[id_name]
    
    #name1 = name_train[len(name_train)-5: len(name_train)]
    name1 = name_train
    model = Transformer(device,config.N,config.num_class)

    model = model.to(device)
    if (torch.cuda.device_count() > 1)and(dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    


    criterion = nn.CrossEntropyLoss().to(device)

    
    t_total = time.time()
    
    for i in range(len(name1)):
        path = path1+name1[i]+path2
        mat = np.load(path)
        data1c = mat['data']
        data1c = data1c[:,np.newaxis,:,:248]
        label1c = mat['label']
        Data0 = data1c[label1c==0,:]
        Data1 = np.repeat(data1c[label1c==1,:],config.sample,axis=0)
        rd_index = np.random.permutation(Data0.shape[0])
        data_0_downsampled = Data0[rd_index[:Data1.shape[0]],:]
        train_data_p = np.concatenate((Data1,data_0_downsampled),axis=0)
        train_label_p = np.concatenate((np.ones(Data1.shape[0]),np.zeros(data_0_downsampled.shape[0])),axis=0)
        if (i == 0):
            train_datac = train_data_p
            train_labelc = train_label_p
        else:
            train_datac = np.append(train_datac, train_data_p, axis = 0)
            train_labelc = np.append(train_labelc, train_label_p)
        if (dist.get_rank() == 0):
            print(train_datac.shape)
        
            
    datas = train_datac 
    label = train_labelc
    a = np.random.permutation(1120*len(name1)*config.sample)
    datas = datas[a]  
    label = label[a]
        
    num_val = config.val
    val_data = torch.from_numpy(datas[datas.shape[0]-num_val:])
    val_label = torch.from_numpy(label[datas.shape[0]-num_val:])
    datas = datas[:datas.shape[0]-num_val]
    label = label[:label.shape[0]-num_val]
    train_data = torch.from_numpy(datas)
    train_label = torch.from_numpy(label)
    EEG_dataset1 = torch.utils.data.TensorDataset(train_data,train_label)
    EEG_dataset2 = torch.utils.data.TensorDataset(val_data,val_label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(EEG_dataset1)
    nw = min([os.cpu_count(), config.batchsize if config.batchsize > 1 else 0, 8])
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=config.batchsize,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=config.batchsize, shuffle=True, num_workers=nw)


    #############################Train################################
    step = config.lr
    if (dist.get_rank() == 0):
        print(name_test[0])
    val_max = 0
    stepp_new = 0
    
    for i in range(epochs):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%40 == 0 and i>0):
            step = step*0.8
        optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=0.01)
        train_l_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii1, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            cor_inputs = Corr(inputs)
            cor_inputs = cor_inputs.to(device)
            outputs_label, Ttoken, Ftoken = model(inputs,cor_inputs)
            loss = criterion(outputs_label, labels.long()) #+ config.kl*KLloss(Ftoken,Ttoken)
            loss.backward()
            optimizer.step()
            
            train_l_sum += loss.cpu().item()
            train_acc_sum += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(outputs_label, labels)
            acc1_sum += num_acc1
            acc0_sum += num_acc0
            sum_1 += num_1
            n += labels.shape[0]

        sum_0 = n - sum_1
        train_l_sum = train_l_sum / (ii1+1)
        BN = train_acc_sum / n
        acc1 = acc1_sum/sum_1
        acc0 = acc0_sum/sum_0
        
        #Validation
        val_l_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii2, data in enumerate(valloader, 0):
            val_inputs, val_labels = data
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.to(device)
            val_cor_inputs = Corr(val_inputs)
            val_cor_inputs = val_cor_inputs.to(device)
            val_output, Ttoken, Ftoken = model(val_inputs,val_cor_inputs)
            loss_val = criterion(val_output, val_labels.long()) #+ config.kl*KLloss(Ftoken,Ttoken)

            val_l_sum += loss_val.cpu().item()
            val_acc_sum += (val_output.argmax(dim=1) == val_labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(val_output, val_labels)
            val_acc1_sum += num_acc1
            val_acc0_sum += num_acc0
            sum_1 += num_1
            n += val_labels.shape[0]

        sum_0 = n - sum_1
        val_l_sum = val_l_sum / (ii2+1)
        val_BN = val_acc_sum / n
        val_acc1 = val_acc1_sum/sum_1
        val_acc0 = val_acc0_sum/sum_0
            
        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                   "acc1= {:.4f}".format(acc1),
                    "acc0= {:.4f}".format(acc0),
                    "BN= {:.4f}".format(BN),
                    'loss_val: {:.4f}'.format(val_l_sum),
                    "val_BN= {:.4f}".format(val_BN),
                    "val_acc1= {:.4f}".format(val_acc1),
                    "val_acc0= {:.4f}".format(val_acc0),
                    "time: {:.4f}s".format(time.time() - t))
        
        if (val_BN>val_max):
            val_max = val_BN
            stepp_new = 0
            if (dist.get_rank() == 0):
                torch.save(model.state_dict(), config.save)
            
        stepp_new = stepp_new + 1
        if (stepp_new==config.patience):
            break
    dist.barrier()
    if (dist.get_rank() == 0):
    
        print('Finished Training')
        model.load_state_dict(torch.load(config.save,map_location=device))
    
    ###################################Test#########################################
    
    path = path1+name_test[0]+path2
    mat = np.load(path)
    datas = mat['data']
    datas = datas[:,np.newaxis,:,:248]
    label = mat['label']
        
    #Shuffle
    test_data = torch.from_numpy(datas)
    test_label = torch.from_numpy(label)
    EEG_dataset3 = torch.utils.data.TensorDataset(test_data,test_label)
    testloader = torch.utils.data.DataLoader(EEG_dataset3, batch_size=64, shuffle=True)
        
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
        
    for j, data in enumerate(testloader, 0):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        cor_inputs = Corr(inputs)
        cor_inputs = cor_inputs.to(device)
        model.eval()
        outputs_label,_,_ = model(inputs,cor_inputs)
        preds = outputs_label.max(1)[1].type_as(labels)
        al = labels.shape[0]
        TP = 0   
        FP = 0.001   
        TN = 0  
        FN = 0.001  
        for i in range(al):
            if ((preds[i]==1)and(labels[i]==1)):
                TP += 1
            if ((preds[i]==1)and(labels[i]==0)):
                FP += 1
            if ((preds[i]==0)and(labels[i]==1)):
                FN += 1
            if ((preds[i]==0)and(labels[i]==0)):
                TN +=1
        correct = preds.eq(labels).double()
        correct = correct.sum()
        acc_test = correct / len(labels)
        TP_all = TP+TP_all
        TN_all = TN+TN_all
        FP_all = FP+FP_all
        FN_all = FN+FN_all
        
    acc1 = TP_all/(TP_all+FN_all)
    acc0 = TN_all/(TN_all+FP_all)
    BN_all = (acc1+acc0)/2
    BNmatrix[id_name] = BN_all
    PRmatrix[id_name,0] = acc1
    PRmatrix[id_name,1] = acc0
    if (dist.get_rank() == 0):
        print(name_test[0]," Test set results:","acc1= {:.4f}".format(acc1),"acc0= {:.4f}".format(acc0),"BN= {:.4f}".format(BN_all))
        print(BNmatrix)
        print(PRmatrix)
    dist.barrier()

name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']
BNmatrix = BNmatrix*100
acc = np.mean(BNmatrix)
var = np.var(BNmatrix)
std = np.sqrt(var)
std = std

if (dist.get_rank() == 0):
    print(BNmatrix)
    print(PRmatrix*100)
    print(acc, "+-", std)
    print(np.mean(PRmatrix*100, axis=0))

#python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 TFF-Former-RSVP/run.py