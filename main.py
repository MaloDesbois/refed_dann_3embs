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
from utils_refed_dann import SupervisedContrastiveLoss, evaluation  
from model import ConvTranDisentangle





def global_loop(data_source,epochs):
        nom=''
        for data_set in data_source:
            a0=rep_geo[f'{data_set}']
            
            nom+=f'+{a0}'
        train_dataloader,data_shape_source,dates=data_loading_source(data_source)
        
        n_classes=11
        dim_ff=64
        data_shape=(data_shape_source[0],data_shape_source[2],data_shape_source[1])
        config={'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':dim_ff}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvTranDisentangle(config,n_classes,dates).to(device)



        learning_rate = 0.0001
        loss_fn = nn.CrossEntropyLoss()
        scl = SupervisedContrastiveLoss()

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


        valid_f1 = 0.
        
        

        for epoch in range(epochs):
            start = time.time()
            model.train()
            tot_loss = 0.0
            domain_loss = 0.0
            contra_tot_loss = 0.0
            den = 0

            for xm_batch, y_batch, dom_batch in train_dataloader:
                if xm_batch.shape[0] != 64:
                    continue

                x_batch,m_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
                x_batch = x_batch.to(device)
                m_batch = m_batch.to(device)
                y_batch = y_batch.to(device)
                dom_batch = dom_batch.to(device)
                optimizer.zero_grad()
                _, inv_emb, inf_emb,irr_emb, pred_inf,pred_irr,adv_pred,pred = model(x_batch,m_batch)

                ohe_label = F.one_hot(y_batch,num_classes=n_classes).cpu().detach().numpy()
                ohe_dom = F.one_hot(dom_batch,num_classes=len(data_source)).cpu().detach().numpy()

                ##### DOMAIN CLASSIFICATION #####
                loss_ce_inf_dom = loss_fn(pred_inf, dom_batch)
                loss_ce_irr_dom = loss_fn(pred_irr, dom_batch)

                ##### MIXED MAINFOLD & CONTRASTIVE LEARNING ####

                cl_labels_npy = y_batch.cpu().detach().numpy()
                y_mix_labels = np.concatenate([ cl_labels_npy , cl_labels_npy],axis=0)

                #DOMAIN LABEL FOR DOMAIN-CLASS SPECIFIC EMBEDDING and DOMAIN SPECIFIC EMBEDDING IS 0 OR 1
                spec_dc_dom_labels = dom_batch.cpu().detach().numpy()
                #DOMAIN LABEL FOR INV EMBEDDING IS 8 IF 8 DATASETS 
                inv_dom_labels = np.ones_like(spec_dc_dom_labels) * 8

                dom_mix_labels = np.concatenate([inv_dom_labels, spec_dc_dom_labels],axis=0)
                #joint_embedding = torch.concat([inv_emb, spec_emb_d])

                #mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)# k*(d+1) 
                #mixdl_loss_supContraLoss = sup_contra_Cplus2_classes(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
                inv_emb_norm =  nn.functional.normalize(inv_emb)
                
                inf_emb_norm = nn.functional.normalize(inf_emb)
                irr_emb_norm = nn.functional.normalize(irr_emb)
                
                ortho_loss1 = torch.mean(torch.sum(inv_emb_norm*inf_emb_norm,dim=1))
                ortho_loss2 = torch.mean(torch.sum(irr_emb_norm*inf_emb_norm,dim=1))
                
                
                adv_loss = loss_fn(adv_pred,dom_batch) 
                




                #contra_loss = mixdl_loss_supContraLoss

                ####################################

                loss = loss_fn(pred, y_batch) + ortho_loss1 + ortho_loss2+adv_loss+loss_ce_irr_dom+loss_ce_inf_dom

                loss.backward() # backward pass: backpropagate the prediction loss
                optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
                
                den+=1.


            end = time.time()
            #pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
            #f1_val = f1_score(labels_valid, pred_valid, average="weighted")
            #if f1_val > valid_f1:
            torch.save(model.state_dict(), f"model_REFeD+dann_3emb{nom}.pth")
                #valid_f1 = f1_val
                #pred_test, labels_test = evaluation(model, test_dataloader, device)
                #f1 = f1_score(labels_test, pred_test, average="weighted")
                
            print(" at Epoch %d: training time %d"%(epoch+1, (end-start)))
                #print(confusion_matrix(labels_test, pred_test))
            #else:
             #   print("TOT AND CONTRA AND TRAIN LOSS at Epoch %d: %.4f %.4f"%(epoch+1, tot_loss/den, contra_tot_loss/den))
            #sys.stdout.flush()
        return model



def test_loop(modèle,data_target):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      test_dataloader,target_data,_=data_loading_target(data_target)
      pred_test, labels_test = evaluation(modèle, test_dataloader, device)
      f1 = f1_score(labels_test, pred_test, average="weighted")
      return f1




def final_test(listes_données,epochs):
  dict_reda_3emb={}
  rep_geo={f'{R2018}':'R18',f'{R2019}':'R19',f'{R2020}':'R20',f'{T2018}':'T18',f'{T2019}':'T19',f'{T2020}':'T20',f'{L2018}':'L18',f'{L2019}':'L19',f'{L2020}':'L20'}
  jeu_test=list(combinations(listes_données,8))
  #jeu_filtré=[elem for elem in jeu_test if rep_geo[f'{elem[0]}'][0]==rep_geo[f'{elem[1]}'][0]  ]
  #ensemble=[(elem[0],elem[1],test) for elem,test in product(jeu_filtré,listes_données)]# if rep_geo[f'{test}'] not in [rep_geo[f'{elem[0]}'],rep_geo[f'{elem[1]}'] ]]
  #dbg=[(rep_geo[f'{ens[0]}'],rep_geo[f'{ens[1]}'],rep_geo[f'{ens[2]}'])for ens in ensemble if  rep_geo[f'{ens[2]}'] not in [rep_geo[f'{ens[0]}'],rep_geo[f'{ens[1]}']] ]
  #for dd in dbg:
  #    print(dd)
  #return (dbg) Pour vérifier que ça fonctionne correctement
  for i in range(len(listes_données)):
      jeu = listes_données[:i]+listes_données[i+1:]
      test= listes_données[i]
  

    
    #a2=f'{jeu[2]}'
    
    
      modèle=global_loop(jeu,epochs)
    
      a2=f'{test}'
      dict_reda_3emb[f'test {rep_geo[a2]}']= test_loop(modèle,test)
      with open('dict_reda_3emb','wb') as f :
        pickle.dump(dict_reda_3emb,f)
  return dict_reda_3emb

liste_data=[R2018,R2019,R2020,L2018,L2019,L2020,T2018,T2019,T2020]

dict_refed_8=final_test(liste_data,100)
print(dict_refed_8)
