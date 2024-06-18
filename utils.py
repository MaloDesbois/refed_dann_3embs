import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,Dataset,ConcatDataset,Sampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from einops import rearrange
import time
import random
import math
import pickle
from itertools import combinations
import torch.nn.functional as F
from torch.autograd import Function


#### pour avoir les données

L2018 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\l2018_modif.npz', allow_pickle=True)
L2019 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\l2019_modif.npz', allow_pickle=True)
L2020 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\l2020_modif.npz', allow_pickle=True)
R2018 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\r2018_modif.npz', allow_pickle=True)
R2019 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\r2019_modif.npz', allow_pickle=True)
R2020 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\r2020_modif.npz', allow_pickle=True)
T2018 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\t2018_modif.npz', allow_pickle=True)
T2019 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\t2019_modif.npz', allow_pickle=True)
T2020 = np.load(r'C:\Users\stginrae\Documents\GitHub\refed_dann_3embs\Data\t2020_modif.npz', allow_pickle=True)
rep_geo={f'{R2018}':'R18',f'{R2019}':'R19',f'{R2020}':'R20',f'{T2018}':'T18',f'{T2019}':'T19',f'{T2020}':'T20',f'{L2018}':'L18',f'{L2019}':'L19',f'{L2020}':'L20'}
####

## preparation data

def prep_data_(data,ratio=0,ratio_supprimé=0): # datas doit être une liste contenant 1 ou 2 jeux de donnés

    mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10}


    data,msk=suppr(data,ratio_supprimé) # peut être utiliser si l'on souhaite diminuer la quantité de points d'acquisition dans les données
    data,_,mask=comp(data,msk) # rempli les données pour les mettre au fromat 365 j et donne le mask correspondant aux jours où on a mit un 0
    values=data['X_SAR']
    data_shape=data['X_SAR'].shape
    dates=data['dates_SAR']




    labels=data['y']
    labels=[mapping[v] if v in mapping else v for v in labels ]

    max_values = np.percentile(values,99)
    min_values = np.percentile(values,1)
    values_norm=(values-min_values)/(max_values-min_values)
    values_norm[values_norm>1] = 1
    values_norm[values_norm<0] = 0
    values = values_norm                                      # les données sont normalisées
    values=add_mask(values,mask)                              # ajoute le mask aux données pour qu'il soit disponiuble dans le dataloader

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indice = sss.split(values,labels)

    tv_index, test_index = next(indice)

    values_tv=[]
    values_test=[]
    labels_tv=[]
    labels_test=[]
    for i in tv_index :
        values_tv+=[values[i]]
        labels_tv+=[labels[i]]
    for j in test_index :
        values_test+=[values[j]]
        labels_test+=[labels[j]]


    sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=0)
    indice2=sss2.split(values_tv,labels_tv)
    train_index,validation_index = next(indice2)

    values_train=[]
    values_validation=[]
    labels_train=[]
    labels_validation=[]

    for i in train_index :
        values_train+=[values_tv[i]]
        labels_train+=[labels_tv[i]]
    for j in validation_index :
        values_validation += [values_tv[j]]
        labels_validation += [labels_tv[j]]


    values_train=np.array(values_train)
    values_validation=np.array(values_validation)
    values_test=np.array(values_test)
    labels_train=np.array(labels_train)
    labels_validation=np.array(labels_validation)
    labels_test=np.array(labels_test)







    return values_train,values_validation,values_test,labels_train,labels_validation,labels_test,dates,data_shape



def data_loading_source(data_source):
        values_train = []
        labels_train = []
        labels_domain_train = []
        
        for i, data in enumerate(data_source):
            values_train_source, _, _, labels_train_source, _, _, dates, data_shape = prep_data_(data)
            
            values_train.append(values_train_source)
            labels_train.append(labels_train_source)
            labels_domain_train.append(np.ones(labels_train_source.shape[0])*i)

        # Convert lists to numpy arrays after concatenation
        values_train = np.concatenate(values_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        labels_domain_train = np.concatenate(labels_domain_train, axis=0)



        x_train=torch.tensor(values_train,dtype=torch.float32)
        y_train=torch.tensor(labels_train,dtype=torch.int64)
        dom_train=torch.tensor(labels_domain_train,dtype=torch.int64)




        train_dataset = TensorDataset(x_train, y_train, dom_train)


        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)


        

        return train_dataloader,data_shape,dates
def data_loading_target(data_target):
      values_train_target,values_validation_target,values_test_target,labels_train_target,labels_validation_target,labels_test_target,dates,data_shape_target=prep_data_(data_target)
      values_test=values_test_target
      labels_test=labels_test_target
      x_test=torch.tensor(values_test,dtype=torch.float32)
      y_test=torch.tensor(labels_test,dtype=torch.int64)
      test_dataset = TensorDataset(x_test, y_test)
      test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)
      target_data=rep_geo[f'{data_target}']
      return test_dataloader,target_data,dates

##divers


class customdata(Dataset):
  def __init__(self,values,labels):
    self.values=values
    self.labels=labels
  def __len__(self):
    return len(self.values)
  def  shape(self):
    return self.values.shape
  def __getitem__(self,id):
    value=self.values[id]
    label=self.labels[id]
    return value, label


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif -val_loss > -(self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print("Saved new best model.")
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False



def get_day_count(dates,ref_day='09-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int) #(dates - ref_day).astype('timedelta64[D]').astype(int)#
    return torch.tensor(days_elapsed,dtype=torch.long)

def add_mask(values,mask): # permet d'attacher les mask aux données pour pouvoir faire les batchs sans perdre le mask
    mask=mask.unsqueeze(0).unsqueeze(-1)
    shape=values.shape
    mask=mask.expand(shape[0],-1,-1)
    values=torch.tensor(values,dtype=torch.float32)

    valuesWmask=torch.cat((values,mask),dim=-1)
    return valuesWmask

def comp (data,msk) : #permet de formater les données avec 365 points d'acquisitions
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  j_p=(data['dates_SAR']-ref).astype('timedelta64[D]').astype(int)
  année=list(range(365))

  année = [ref + np.timedelta64(j, 'D') for j in année ]
  mask = []

  for i,jour in enumerate(année):
    if jour not in data['dates_SAR']:

      mask+=[0]
      msk=np.insert(msk,i,0)
      data_r['dates_SAR']=np.insert(data_r['dates_SAR'],i,jour)
      data_r['X_SAR']=np.insert(data_r['X_SAR'],i,[0,0],axis=1)
    else:
      mask+=[1]


  mask=torch.tensor(mask,dtype=torch.float32)
  msk=torch.tensor(msk,dtype=torch.float32)
  return data_r,mask,msk


def suppr (data,ratio):
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  nbr,seq_len,channels=data['X_SAR'].shape #(nbr,seq_len,channels)
  
  nbr_indice=int(seq_len*ratio)
  indice=list(range(seq_len))
  indice=random.sample(indice,nbr_indice)
  mask=[0 if i in indice else 1 for i in range(seq_len)]
  mask=torch.tensor(mask)

  data_r['X_SAR']=torch.tensor(data_r['X_SAR'])
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].masked_fill(mask==0,0)
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].numpy()
  mask=mask.numpy()
  return data_r,mask
