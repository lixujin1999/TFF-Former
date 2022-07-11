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
from scipy.io import loadmat
from sklearn import preprocessing
from scipy import signal

from config import config
from model import Transformer
from metric import GELU, LabelSmoothingLoss, LabelSmoothingLoss1, acc10, Corr, att_norm, accuracy, Window, stand, filter

#GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

dist.init_process_group(backend="nccl")#, init_method="env://", world_size=torch.cuda.device_count(),rank=local_rank)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']
path1 = os.path.abspath(os.curdir)+'/TFF-Former-SSVEP/benchmark/'
path2 = '.mat'

epochs = config.epoch

PRmatrix = np.zeros(len(name))
rd_index = np.random.permutation(248)
rd_sort = np.sort(rd_index[:248])
index_class = range(0,40)

for id_name in range(len(name)):
    name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']
    name_test = [name[id_name]]
    name_train = name
    del name_train[id_name]
    
    #name1 = name_train[len(name_train)-5: len(name_train)]
    name1 = name_train
    model = Transformer(device,config.N,config.num_class)
    '''
    #GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    #GPUs
    '''
    model = model.to(device)
    if (torch.cuda.device_count() > 1)and(dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    criterion = LabelSmoothingLoss()
    
    t_total = time.time()
    
    for i in range(len(name1)):
        path = path1+name1[i]+path2
        mat = loadmat(path)
        data1c = mat['data']
        data1c = data1c[:,150:150+config.T,index_class,:]
        #data1c = data1c[index_C,150:150+config.T,:config.num_class,:]
        data1c = data1c.transpose(2,3,0,1)#label*block*C*T
        train_data_p = np.zeros((6*config.num_class,data1c.shape[2],data1c.shape[3]))
        train_label_p = np.zeros(6*config.num_class)
        for j in range(config.num_class):
            train_data_p[6*j:6*j+6] = data1c[j]
            train_label_p[6*j:6*j+6] = np.ones(6)*j

        if (i == 0):
            train_datac = train_data_p
            train_labelc = train_label_p
        else:
            train_datac = np.append(train_datac, train_data_p, axis = 0)
            train_labelc = np.append(train_labelc, train_label_p)
        if (dist.get_rank() == 0):
            print(train_datac.shape)
        
        
    if (dist.get_rank() == 0):
        print('Start filtering')
    datas = train_datac 
    datas = filter(datas)  
    datas = stand(datas)
    datas = datas[:,np.newaxis,:,:]
    label = train_labelc

    if (dist.get_rank() == 0):
        print(datas.shape)
    a = np.random.permutation(datas.shape[0])
    datas = datas[a]  
    label = label[a]
        
    num_val = config.val
    val_data = torch.FloatTensor(datas[datas.shape[0]-num_val:])
    val_label = torch.FloatTensor(label[datas.shape[0]-num_val:])
    datas = datas[:datas.shape[0]-num_val]
    label = label[:label.shape[0]-num_val]
    train_data = torch.FloatTensor(datas)
    train_label = torch.FloatTensor(label)
    EEG_dataset1 = torch.utils.data.TensorDataset(train_data,train_label)
    EEG_dataset2 = torch.utils.data.TensorDataset(val_data,val_label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(EEG_dataset1)
    nw = min([os.cpu_count(), config.batchsize if config.batchsize > 1 else 0, 8])
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=config.batchsize,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=config.batchsize, shuffle=True, num_workers=nw)
    
    ##########################Train###############################
    step = config.lr
    if (dist.get_rank() == 0):
        print(name_test[0])
    val_max = 0
    stepp_new = 0
    
    for i in range(epochs):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%20 == 0 and i>0):
            step = step*0.8
        optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=0.01)
        train_l_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii1, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            cor_inputs = Corr(inputs).to(device)
            outputs_label = model(inputs,cor_inputs)
            loss = criterion(outputs_label, labels.long())
            loss.backward()
            optimizer.step()
            
            train_l_sum += loss.cpu().item()
            train_acc_sum += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
            n += labels.shape[0]

        sum_0 = n - sum_1
        train_l_sum = train_l_sum / (ii1+1)
        BN = train_acc_sum / n
        
        #Validation
        val_l_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii2, data in enumerate(valloader, 0):
            val_inputs, val_labels = data
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.to(device)
            val_cor_inputs = Corr(val_inputs).to(device)
            val_output = model(val_inputs,val_cor_inputs)
            loss_val = criterion(val_output, val_labels.long())

            val_l_sum += loss_val.cpu().item()
            val_acc_sum += (val_output.argmax(dim=1) == val_labels).sum().cpu().item()
            n += val_labels.shape[0]

        sum_0 = n - sum_1
        val_l_sum = val_l_sum / (ii2+1)
        val_BN = val_acc_sum / n
            
        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                    "BN= {:.4f}".format(BN),
                    'loss_val: {:.4f}'.format(val_l_sum),
                    "val_BN= {:.4f}".format(val_BN),
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
    
    ############################Test#####################################
    # Testing
    path = path1+name_test[0]+path2
    mat = loadmat(path)
    data1c = mat['data']
    data1c = data1c[:,150:150+config.T,index_class,:]
    #data1c = data1c[index_C,150:150+config.T,:config.num_class,:]
    data1c = data1c.transpose(2,3,0,1)#label*block*C*T
    test_data_p = np.zeros((config.num_class*6,data1c.shape[2],data1c.shape[3]))
    test_label_p = np.zeros(config.num_class*6)
    for j in range(config.num_class):
        test_data_p[6*j:6*j+6] = data1c[j]
        test_label_p[6*j:6*j+6] = np.ones(6)*j

    datas = test_data_p
    datas = filter(datas)
    datas = stand(datas)
    if (config.T==250):
        datas = datas[:,np.newaxis,:,rd_sort]
    else:
        datas = datas[:,np.newaxis,:,:]
    label = test_label_p
        
    
    test_data = torch.FloatTensor(datas)
    test_label = torch.FloatTensor(label)
    EEG_dataset3 = torch.utils.data.TensorDataset(test_data,test_label)
    testloader = torch.utils.data.DataLoader(EEG_dataset3, batch_size=240, shuffle=True)
    
    correct = 0
    total = 0
    for j, data in enumerate(testloader, 0):
        inputs, labels = data
        model.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)
        cor_inputs = Corr(inputs).to(device)
        outputs_label = model(inputs,cor_inputs)
        correct += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
        total += labels.size(0)

    acc_test = correct/ total

    PRmatrix[id_name] = acc_test
    if (dist.get_rank() == 0):
        print(name_test[0]," Test set results:","Accuracy= {:.4f}".format(acc_test))
        print(PRmatrix*100)
    dist.barrier()

name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35']
PRmatrix = PRmatrix*100
acc = np.mean(PRmatrix)
var = np.var(PRmatrix)
std = np.sqrt(var)
std = std

if (dist.get_rank() == 0):
    print(PRmatrix)
    print(acc, "+-", std)

#python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-SSVEP/run.py
