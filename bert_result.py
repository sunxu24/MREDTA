import math
import os
import time
import re
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from transformers import AutoTokenizer, AutoModel
import DataHelper as DH

import emetrics as EM
import json
import sys
import joblib

Drug_BERT_path = "./pretrained_BERTmodel/biobert-v1.1"
Target_BERT_path = "./pretrained_BERTmodel/prot_bert_bfd"

def seed_torch(seed=2):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, label):
        self.texta = texta
        self.textb = textb
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.label[item]

    def __len__(self):
        return len(self.texta)

def BatchPad_data(batch_data, pad=0):
    # batch_data [32,xx,768]
    # get max_len
    max_len=0
    for item in batch_data:
        xx = item.size(1)
        if xx > max_len:
            max_len = xx
    target_size = (1, max_len, 768)
    # padding
    padding_data=[]
    for item in batch_data:
        padding_size = (target_size[1] - item.size(1))
        if padding_size>0:
            padding_tensor = torch.zeros(1, padding_size, 768).cuda()
            return_tensor = torch.cat((item, padding_tensor), dim=1)
        else:
            return_tensor = item
        padding_data.append(return_tensor)

    padding_data = torch.stack(padding_data,dim=0)
    # print("after padding shape")
    # print(padding_data.shape)
    return padding_data

def BatchPad(batch_data, pad=0):
    # batch_data = attention mask or input
    
    # get max_len_a
    max_len_a=0
    for item in batch_data:
        xx = item.size(1)
        if xx > max_len_a:
            max_len_a = xx
    # print("max_len_a")
    # print(max_len_a)
    target_size = (1, max_len_a)
    
    # padding
    padding_data=[]
    for item in batch_data:
        padding_size = (target_size[1] - item.size(1))
        if padding_size>0:
            padding_tensor = torch.zeros(1, padding_size).cuda()
            return_tensor = torch.cat((item, padding_tensor), dim=1)
        else:
            return_tensor = item
        padding_data.append(return_tensor)

    padding_data = torch.stack(padding_data,dim=0)
    # print("after padding shape")
    # print(padding_data.shape)
    return padding_data

if __name__ == '__main__':
    
    '''
    # kiba dataset from DeepDTA
    smile_maxlenKB, proSeq_maxlenKB = 100, 1000
    trainKB_num, testKB_num = 98545, 19709
    fpath_kiba = './data_kiba'
    
    drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)
    # affiMatrix 得到非空值的list，drug和target的非空值都扩张到118254
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('kiba', drug, target, affinity)
    # shuttle
    Drugs_shuttle, Targets_shuttle, affiMatrix_shuttle \
                                = DH.Shuttle(drug_seqs, target_seqs, affiMatrix)
    print(f"Drug_sample: {Drugs_shuttle[0]}, {Drugs_shuttle[1]}")
    print(f"Target_sample: {Targets_shuttle[0]}, {Targets_shuttle[1]}")
    Drugs_fold1 = Drugs_shuttle[0:19709]
    Targets_fold1 = Targets_shuttle[0:19709]
    affiMatrix_fold1 = affiMatrix_shuttle[0:19709]

    Drugs_fold2 = Drugs_shuttle[19709:39418]
    Targets_fold2 = Targets_shuttle[19709:39418]
    affiMatrix_fold2 =affiMatrix_shuttle[19709:39418]

    Drugs_fold3 = Drugs_shuttle[39418:59127]
    Targets_fold3 = Targets_shuttle[39418:59127]
    affiMatrix_fold3 =affiMatrix_shuttle[39418:59127]

    Drugs_fold4 = Drugs_shuttle[59127:78836]
    Targets_fold4 = Targets_shuttle[59127:78836]
    affiMatrix_fold4 =affiMatrix_shuttle[59127:78836]

    Drugs_fold5 = Drugs_shuttle[78836:98545]
    Targets_fold5 = Targets_shuttle[78836:98545]
    affiMatrix_fold5 =affiMatrix_shuttle[78836:98545]

    Drugs_fold6 = Drugs_shuttle[98545:118254]
    Targets_fold6 = Targets_shuttle[98545:118254]
    affiMatrix_fold6 = affiMatrix_shuttle[98545:118254]
    
    train1_drugs = np.hstack((Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5))
    train1_targets = np.hstack((Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5))
    train1_affinity = np.hstack((affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5))
    
    '''
    #98545
    
    '''
    # davis dataset from DeepDTA len = 30056
    smile_maxlenDA, proSeq_maxlenDA = 85, 1200
    trainDV_num, testDV_num = 25046, 5010
    fpath_davis = './data_davis'
    drug, target, affinity = DH.LoadData(fpath_davis, logspance_trans=True)
    
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('davis', drug, target, affinity)
    
    # shuffle
    Drugs_shuttle, Targets_shuttle, affiMatrix_shuttle \
                                = DH.Shuttle(drug_seqs, target_seqs, affiMatrix)
    print(f"Drug_sample: {Drugs_shuttle[0]}, {Drugs_shuttle[1]}")
    print(f"Target_sample: {Targets_shuttle[0]}, {Targets_shuttle[1]}")
    # davis train data 5-fold
    Drugs_fold1 = Drugs_shuttle[0:5010]
    Targets_fold1 = Targets_shuttle[0:5010]
    affiMatrix_fold1 = affiMatrix_shuttle[0:5010]

    Drugs_fold2 = Drugs_shuttle[5010:10020]
    Targets_fold2 = Targets_shuttle[5010:10020]
    affiMatrix_fold2 =affiMatrix_shuttle[5010:10020]

    Drugs_fold3 = Drugs_shuttle[10020:15030]
    Targets_fold3 = Targets_shuttle[10020:15030]
    affiMatrix_fold3 =affiMatrix_shuttle[10020:15030]

    Drugs_fold4 = Drugs_shuttle[15030:20040]
    Targets_fold4 = Targets_shuttle[15030:20040]
    affiMatrix_fold4 =affiMatrix_shuttle[15030:20040]

    Drugs_fold5 = Drugs_shuttle[20040:25046]
    Targets_fold5 = Targets_shuttle[20040:25046]
    affiMatrix_fold5 =affiMatrix_shuttle[20040:25046]

    Drugs_fold6 = Drugs_shuttle[25046:30056]
    Targets_fold6 = Targets_shuttle[25046:30056]
    affiMatrix_fold6 = affiMatrix_shuttle[25046:30056]
    
    train1_drugs = np.hstack((Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5))
    train1_targets = np.hstack((Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5))
    train1_affinity = np.hstack((affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5))
    '''
    
    
    # EGFR dataset:2091
    # new EGFR:2111
    # smile_maxlenDA, proSeq_maxlenDA = 2091, 1200
    # trainDV_num, testDV_num = 25046, 5010
    batch_size=64
    fpath_EGFR = './data_qincaisu'
    drug, target, affinity = DH.LoadData(fpath_EGFR, logspance_trans=False)
    print(f"len(drug)={len(drug)}")
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('EGFR', drug, target, affinity)
    print(f"len(drug_seqs)={len(drug_seqs)}")
    print(f"Drug_sample: {drug_seqs[0]}, {drug_seqs[1]}")
    print(f"Target_sample: {target_seqs[0]}")
    
    test1_drugs, test1_targets,  test1_affinity= drug_seqs, target_seqs, affiMatrix
    test_iter = DatasetIterater(test1_drugs, test1_targets, test1_affinity)
    test_loader = Data.DataLoader(test_iter, batch_size, False)
    
    for count in range(1):   
        # pretrained_weights = torch.load(model_fromTrain1) # 加载预训练模型
        # model.load_state_dict(pretrained_weights)
        # EPOCHS, batch_size, accumulation_steps = 600, 64, 32 
        
        
        # train_iter = DatasetIterater(train1_drugs, train1_targets, train1_affinity)
        # val_iter = DatasetIterater(Drugs_fold6, Targets_fold6, affiMatrix_fold6)
        
        # train_loader = Data.DataLoader(train_iter, batch_size, False)
        # val_loader = Data.DataLoader(val_iter, batch_size, False)
        
        tokenizer_target = AutoTokenizer.from_pretrained(Target_BERT_path)
        model_target = AutoModel.from_pretrained(Target_BERT_path)
        tokenizer_drug = AutoTokenizer.from_pretrained(Drug_BERT_path)
        model_drug = AutoModel.from_pretrained(Drug_BERT_path)

        # Drug
        
        seed_torch(seed=2)
        for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(test_loader):
            
            print("第{}轮".format(train_batch_idx))

            if torch.cuda.is_available():
                real_affi = real_affi.cuda()
            
            if torch.cuda.is_available():
                BERT_model = model_drug.cuda()
            else:
                BERT_model = model_drug
            with torch.no_grad():
                bert_results=[]
                bert_inputs=[]
                bert_attentions=[]
                enc_inputs = SeqDrug
                # if(train_batch_idx==30):
                for text in enc_inputs:
                    # print(text)
                    # tokenizer_drug.clear_cache()
                    inputs = tokenizer_drug.encode_plus(
                        text,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    # print(type(inputs))
                    # input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
                    input_ids = inputs['input_ids'].clone().detach()
                    # print(input_ids.shape)
                    # print(input_ids)
                    # attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
                    attention_mask = inputs['attention_mask'].clone().detach()
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                    # 截断序列
                    max_sequence_length = 512
                    if input_ids.shape[1] > max_sequence_length:    
                        input_ids = input_ids[:,:max_sequence_length]
                        attention_mask = attention_mask[:,:max_sequence_length]
                    # tokenizer.encode_plus返回对应单词的编码
                    bert_inputs.append(input_ids)
                    # 将输入传递给模型
                    outputs = BERT_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # 获取所有标记的隐藏表示形式
                    hidden_states = outputs.last_hidden_state
                    # print(hidden_states.shape)
                    # print(hidden_states)
                    # print(hidden_states[0])
                    bert_attentions.append(attention_mask) # [batchsize,1,src_len]
                    bert_results.append(hidden_states) # [batchsize,3,1024]
                        
                
            bert_results = BatchPad_data(bert_results)
            bert_attentions = BatchPad(bert_attentions)
            bert_inputs = BatchPad(bert_inputs)     
            bert_inputs = torch.squeeze(bert_inputs,dim=1)
            bert_inputs = bert_inputs.to(torch.int64)
            
            path_results = "./BERT_data/BERT_results/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_results, path_results)
            path_attentions = "./BERT_data/BERT_attentions/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_attentions, path_attentions)
            path_inputs = "./BERT_data/BERT_inputs/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_inputs, path_inputs)
        print(bert_inputs[0])
        print(bert_inputs[1])
    '''
        # Target
        seed_torch(seed=2)
        for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(val_loader):
            print("第{}轮".format(train_batch_idx))
            if torch.cuda.is_available():
                real_affi = real_affi.cuda()
            if torch.cuda.is_available():
                BERT_model = model_target.cuda()
            with torch.no_grad():
                bert_results=[]
                bert_inputs=[]
                bert_attentions=[]
                enc_inputs = SeqTar
                for text in enc_inputs:
                    text = re.sub(r"[UZOB]", "X", text)
                    # tokenizer_drug.clear_cache()
                    inputs = tokenizer_target.encode_plus(
                        text,
                        return_tensors="pt"
                    )
                    # print(type(inputs))
                    # input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
                    input_ids = inputs['input_ids'].clone().detach()
                    # print(input_ids.shape)
                    print(input_ids)
                    # attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
                    attention_mask = inputs['attention_mask'].clone().detach()
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                    # 截断序列
                    max_sequence_length = 512
                    if input_ids.shape[1] > max_sequence_length:    
                        input_ids = input_ids[:,:max_sequence_length]
                        attention_mask = attention_mask[:,:max_sequence_length]
                    # tokenizer.encode_plus返回对应单词的编码
                    bert_inputs.append(input_ids)
                    # 将输入传递给模型
                    outputs = BERT_model(input_ids=input_ids)
                    
                    # 获取所有标记的隐藏表示形式
                    hidden_states = outputs.last_hidden_state
                    # print(hidden_states.shape)
                    # print(hidden_states)
                    # print(hidden_states[0])
                    bert_attentions.append(attention_mask) # [batchsize,1,src_len]
                    bert_results.append(hidden_states) # [batchsize,3,1024]
                
                
            bert_results = BatchPad_data(bert_results)
            bert_attentions = BatchPad(bert_attentions)
            bert_inputs = BatchPad(bert_inputs)     
            bert_inputs = torch.squeeze(bert_inputs,dim=1)
            bert_inputs = bert_inputs.to(torch.int64)
            
            path_results = "./BERT_data/BERT_results/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_results, path_results)
            path_attentions = "./BERT_data/BERT_attentions/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_attentions, path_attentions)
            path_inputs = "./BERT_data/BERT_inputs/Batch" + str(train_batch_idx) + ".pt"
            torch.save(bert_inputs, path_inputs)
            '''