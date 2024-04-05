import math
import os
import time
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# log_path = "/runs/Nov25_15-04-32_9634a496727a"
# writer = SummaryWriter(log_path)

# Transformer Parameters
# d_model = int(768)
d_model_d = int(768) # Embedding Size for drug
d_model_t = int(1024)
# d_model_t = int(1024) # Embedding Size for target
d_ff = 512 # FeedForward dimension
d_k = d_v = 16 # dimension of K(=Q), V
# d_k = d_v = 16
n_layers = 1 # number of Encoder
n_heads = 4 # number of heads in Multi-Head Attention
# loaded_XT = joblib.load('./BERT features/XT_BERT1.pkl')
# loaded_XD = joblib.load('./BERT features/XD_BERT1.pkl')
Drug_BERT_path = "./pretrained_BERTmodel/biobert-v1.1"
Target_BERT_path = "./pretrained_BERTmodel/prot_bert_bfd"

smile_maxlenKB, proSeq_maxlenKB = 100, 1000
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
seed_torch(seed=2)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # seq_q=seq_k: [batch_size, seq_len]
    # print(seq_q.size())
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask.bool(), -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.d_model = d_model
    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input Q,K,V are all stream0
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        # print("##### input_Q.size() #####")
        # print(input_Q.size())
        ##residual, batch_size = input_Q, input_Q.size(0)
        
        batch_size, seq_len, model_len = input_Q.size()
        # print(input_Q.size())
        residual_2D = input_Q.view(batch_size*seq_len, model_len)
        # print(residual_2D.size())
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                      2) # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                               1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        if torch.cuda.is_available():    
            return nn.LayerNorm(self.d_model).cuda()(output+residual), attn
        return nn.LayerNorm(self.d_model)(output+residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        
        residual = inputs
        output = self.fc(inputs)
        if torch.cuda.is_available():
            return nn.LayerNorm(self.d_model).cuda()(output+residual) # [batch_size, seq_len, d_model]
        return nn.LayerNorm(self.d_model)(output+residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
# Positional Encoding 和 Embedding 和 skip connect

class Encoder_Target(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Encoder_Target, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
    def forward(self, enc_inputs, batch_idx, process):
        labeled_targets = DH.LabelDT(enc_inputs, smile_maxlenKB)
        enc_inputs = torch.IntTensor(labeled_targets)
        if torch.cuda.is_available():
            enc_inputs = enc_inputs.cuda()
        # process: train/val
        #enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)# [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
    
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]

        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)
        # print("####### stream0 #######")
        
        #skip connect -> stream0
        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2

class Encoder_Drug(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Encoder_Drug, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        # self.item = item
        self.stream0 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.dropout = torch.nn.Dropout(p=0.5)
        self.stream1 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.d_model = d_model
    def forward(self, enc_inputs, tokenizer, BERT_model, batch_idx, key, process, dataset, batch_size):
        # key: Drug or Target
        # process: train or val
        # batch_size = 256
        bert_inputs_path = "./BERT_data_can/BERT_data_"+ str(batch_size)+"/BERT_" + dataset + "/BERT_" + key + "_" + process + "/BERT_inputs/Batch" + str(batch_idx) + ".pt" 
        bert_inputs = torch.load(bert_inputs_path)   
        bert_results_path =  "./BERT_data_can/BERT_data_"+ str(batch_size)+"/BERT_" + dataset + "/BERT_" + key + "_" + process + "/BERT_results/Batch" + str(batch_idx) + ".pt" 
        bert_results = torch.load(bert_results_path)   
        bert_attentions_path =  "./BERT_data_can/BERT_data_"+ str(batch_size)+"/BERT_" + dataset +  "/BERT_" + key + "_" + process + "/BERT_attentions/Batch" + str(batch_idx) + ".pt" 
        bert_attentions = torch.load(bert_attentions_path)   
        # print(bert_inputs)
        # print("\n")
        # print(bert_results)
        
        if torch.cuda.is_available():
            bert_results = bert_results.cuda()
            bert_inputs = bert_inputs.cuda()
            # bert_attentions = bert_attentions.cuda()    
        
        stream0 = torch.squeeze(bert_results,dim=1) # [batch_size, src_len, 768]        
        
        # 对label完的input进行embedding   
        bert_inputs = self.src_emb(bert_inputs)
        bert_inputs = self.pos_emb(bert_inputs)
        
        # get attention mask and 查询序列和键序列匹配size
        enc_self_attn_mask = bert_attentions
        seq_q = enc_self_attn_mask
        seq_k = seq_q
        batch_size = seq_q.size(0)
        len_q = seq_q.size(2)
        len_k = seq_k.size(2)
        enc_self_attn_mask = enc_self_attn_mask.expand(batch_size, len_q, len_k)     
        
        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)
        
        # 加入dropout层
        # stream0 = self.dropout(stream0)
        
        # print(stream0.shape)
        # print(bert_inputs.shape)
        #skip connect -> stream0
        stream1 = stream0 + bert_inputs
        stream2 = stream0 + bert_inputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2
        # return torch.cat((stream1, stream2), 2)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # d_model=128
        # self.tokenizer_target = AutoTokenizer.from_pretrained(Target_BERT_path)
        # self.model_target = AutoModel.from_pretrained(Target_BERT_path)
        self.tokenizer_drug = AutoTokenizer.from_pretrained(Drug_BERT_path)
        self.model_drug = AutoModel.from_pretrained(Drug_BERT_path)
    
        self.encoderD = Encoder_Drug(DH.drugSeq_vocabSize,d_model_d)
        self.encoderT = Encoder_Target(DH.targetSeq_vocabSize, d_model_t)
        self.fc0 = nn.Sequential(
            nn.Linear(2*(d_model_d+d_model_t), 8*(d_model_d+d_model_t), bias=False),
            nn.LayerNorm(8*(d_model_d+d_model_t)),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*(d_model_d+d_model_t), 2*(d_model_d+d_model_t), bias=False),
            nn.LayerNorm(2*(d_model_d+d_model_t)),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        # self.fc2 = nn.Linear(2*(d_model_d+d_model_t), 1, bias=False) # fc1
        self.fc2 = nn.Linear(8*(d_model_d+d_model_t), 1, bias=False) # 无fc1

    def forward(self, input_Drugs, input_Tars, batch_idx, process, dataset, batch_size):
        # print("#### Transformer ####")
        # input: [batch_size, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # print(type(input_Drugs[0]))
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs,self.tokenizer_drug,self.model_drug, batch_idx, "Drug", process, dataset, batch_size)
        # print(enc_Drugs.size())
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars, batch_idx, process)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        #fc = enc_Drugs_2D1 + enc_Tars_2D1
        try:
            fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)
        except:
            print("最后一个batch只有1个值，不能用.extend")
            enc_Tars_2D1 = enc_Tars_2D1.unsqueeze(0)
            enc_Drugs_2D1 = enc_Drugs_2D1.unsqueeze(0)
            fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)
        fc0 = self.fc0(fc)
        # fc1 = self.fc1(fc0)
        # fc2 = self.fc2(fc1)
        # 去掉fc1层
        fc2 = self.fc2(fc0)
        affi = fc2.squeeze()

        return affi, enc_attnsD0, enc_attnsT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2

if __name__ == '__main__':
    
    # kiba dataset from DeepDTA
    
    smile_maxlenKB, proSeq_maxlenKB = 100, 1000
    train_num, test_num = 98545, 19709
    fpath_kiba = './data_kiba'
    drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)
    
    # davis dataset from DeepDTA len = 30056
    '''
    smile_maxlenDA, proSeq_maxlenDA = 85, 1200
    train_num, test_num = 25046, 5010
    fpath_davis = './data_davis'
    drug, target, affinity = DH.LoadData(fpath_davis, logspance_trans=True)
    '''
    # affiMatrix 得到非空值的list，drug和target的非空值都扩张到118254
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('kiba', drug, target, affinity)
    # drug_seqs, target_seqs, affiMatrix = DH.GetSamples('davis', drug, target, affinity)
    
    # shuffle
    Drugs_shuttle, Targets_shuttle, affiMatrix_shuttle \
                                = DH.Shuttle(drug_seqs, target_seqs, affiMatrix)
    
    
    # kiba dataset
    
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
    '''
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
    '''
    #98545
    
    train1_drugs = np.hstack((Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5))
    train1_targets = np.hstack((Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5))
    train1_affinity = np.hstack((affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5))
    
    # train2_drugs = np.hstack((Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5, Drugs_fold6))
    # train2_targets = np.hstack((Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5, Targets_fold6))
    # train2_affinity = np.hstack((affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold6))
    
    # print(type(train1_drugs_bert))
    model_fromTrain1 = './models/model_fromTrain.pth'
    MSE, CI, RM2 = [], [], []
    load_model = "./models/load/model_fromTrain.pth"
    # load_model = "./backup_no_fc1_lr4/models/model_fromValCI.pth"
    # load_model="./models_bs64_dk16_ep265_lr3/model_fromValCI.pth"
    # load_model="./results_kiba_iso_bs64/model_fromVal_CI8712_mse177_rm2701.pth"
    for count in range(1):   
        if torch.cuda.is_available():
            model = Transformer().cuda()
        else:
            model = Transformer().cpu()
        
        pretrained_weights = torch.load(load_model) # 加载预训练模型
        model.load_state_dict(pretrained_weights)
        
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # 定义 ReduceLROnPlateau 调度器
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        
        # L2正则化添加
        # optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.01)

        EPOCHS, batch_size, accumulation_steps = 400, 64, 32  # bs=1024 -> update loss
        # EPOCHS, batch_size, accumulation_steps = 300, 16, 32 
        trainEP_loss_list = []
        # valEP_loss_list = []
        min_train_loss = 100000  # save best model in train
        min_val_loss = 100000 # save best model in val

        train_iter = DatasetIterater(train1_drugs, train1_targets, train1_affinity)
        val_iter = DatasetIterater(Drugs_fold6, Targets_fold6, affiMatrix_fold6)
        # train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
        # test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)
        train_loader = Data.DataLoader(train_iter, batch_size, False)
        val_loader = Data.DataLoader(val_iter, batch_size, False)

        dataset="kiba"
        # dataset = "davis"
        '''
        ###############
        ##Train Process
        ###############
        '''
        seed_torch(seed=2)
        
        print("Begin Training!!!!")
        val_mse=[]
        val_ci=[]
        val_RM2=[]
        model_fromValCI = "./models/model_fromValCI.pth"
        model_fromVal = "./models/model_fromVal.pth"
        # 定义早停参数
        best_val_CI = 0
        patience = 15
        counter = 0
        
        for epoch in range(EPOCHS):
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 用于同步 CUDA 操作的函数，它在 GPU 上等待所有之前的 CUDA 操作完成。
            start = time.time()
            model.train() # -> model.eval(), start Batch Normalization and Dropout
            train_sum_loss = 0
            train_obs=[]
            train_pred=[]
            process = "train"
            for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(train_loader):
                # print(SeqDrug.shape) 
                # print(SeqTar.shape) 
                # print(real_affi.shape) # torch.size(32)
                # print(real_affi)

                if torch.cuda.is_available():
                    #     # print(SeqDrug) # str # print(SeqTar) # str # print(real_affi) # tensor float
                    #     SeqDrug, SeqTar, real_affi = np.array(SeqDrug), np.array(SeqTar), np.array(real_affi)
                    #     SeqDrug, SeqTar, real_affi = torch.from_numpy(SeqDrug), torch.from_numpy(SeqTar), torch.from_numpy(real_affi)
                    real_affi = real_affi.cuda()
                else:
                    SeqDrug, SeqTar, real_affi = SeqDrug, SeqTar, real_affi
                
                pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1,enc_attnsD2, enc_attnsT2 \
                                                = model(SeqDrug, SeqTar, train_batch_idx, process, dataset, batch_size) # pre_affi: [batch_affini]
                # print(pre_affi)
                # real_affi = real_affi.float()
                real_affi = real_affi.to(torch.float32)
                train_loss = criterion(pre_affi, real_affi)
                train_obs.extend(real_affi.tolist())
                train_pred.extend(pre_affi.tolist())
                
                train_sum_loss += train_loss.item() # loss -> loss.item(), avoid CUDA out of memory
                train_loss.backward()
                
                # batch_size from 32 -> 1024
                if ((train_batch_idx+1)%accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if ((train_batch_idx+1)%200) == 0:
                    print('Epoch:', '%04d' % (epoch+1), 'loss =', '{:.6f}'.format(train_loss))

                if (train_batch_idx+1) == (train_num//batch_size+1):
                    train_epoch_loss = train_sum_loss / (train_batch_idx+1)
                    trainEP_loss_list.append(train_epoch_loss)
                    train_CI = EM.get_cindex(train_obs, train_pred)
                    train_rm2 = EM.get_rm2(train_obs, train_pred)
                    
                    # if (epoch%10==0):
                    #     train_CI = EM.get_cindex(train_obs, train_pred)
                    #     print('Epoch:', '%04d' % (epoch+1), 'train_CI = ', '{:.6f}'.format(train_CI))
                    #     writer.add_scalar('CI/Train', train_CI, epoch)
                    # train_rm2 = EM.get_rm2(train_obs, train_pred)
                    print('\n')
                    print('Epoch:', '%04d' % (epoch+1), 'train_epoch_loss = ', '{:.6f}'.format(train_epoch_loss))
                    
                    # print('Epoch:', '%04d' % (epoch+1), 'train_RM2 = ', '{:.6f}'.format(train_rm2))
    
                    # save best train model
                    if train_epoch_loss < min_train_loss:
                        min_train_loss = train_epoch_loss
                        
                        if count == 0:
                            # torch.save(model.state_dict(), model_fromTrain1)
                            print('Best model in train1 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain1))            
            # if(epoch%100==0):
            # if((epoch+1)%100==0):
            #     torch.save(model.state_dict(), "./models/"+str(epoch)+".pth")            
            
            
            # 记录train 数据
            writer.add_scalar('Loss/Train', train_epoch_loss, epoch)
            writer.add_scalar('CI/Train', train_CI, epoch)
            writer.add_scalar('rm2/Train', train_rm2, epoch)
            
            # 手动释放内存
            torch.cuda.empty_cache()
            ############ val process ############
            '''
            process = "val"
            
            val_sum_loss = 0
            valEP_loss_list,val_obs, val_pred = [], [], []
            model.eval()
            with torch.no_grad():
                for val_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(val_loader):
    
                    real_affi = real_affi.cuda()
                    pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2 \
                                                                                = model(SeqDrug, SeqTar, val_batch_idx, process, dataset, batch_size)
                    
                    val_loss = criterion(pre_affi, real_affi)
                    val_sum_loss += val_loss.item()  # loss -> loss.item(), avoid CUDA out of memory
    
                    val_obs.extend(real_affi.tolist())
                    val_pred.extend(pre_affi.tolist())
    
                    if (val_batch_idx+1) == (len(val_loader)//batch_size+1):
                        val_epoch_loss = val_sum_loss / (val_batch_idx+1)
                        val_CI = EM.get_cindex(val_obs, val_pred)
                        val_rm2 = EM.get_rm2(val_obs, val_pred)
                        valEP_loss_list.append(val_epoch_loss)
                        val_ci.append(val_CI)
                        val_RM2.append(val_rm2)
                        print('Epoch:', '%04d' % (epoch+1), 'val_epoch_loss = ', '{:.6f}'.format(val_epoch_loss))
                        print('Epoch:', '%04d' % (epoch+1), 'val_CI = ', '{:.6f}'.format(val_CI))
                        print('Epoch:', '%04d' % (epoch+1), 'val_RM2 = ', '{:.6f}'.format(val_rm2))
                       
                        # 在验证集上监测损失，然后调整学习率
                        # scheduler.step(val_loss)
                        # 获取当前学习率
                        # current_lr = optimizer.param_groups[0]["lr"]

                        # 使用 SummaryWriter 记录学习率
                        # writer.add_scalar("Learning Rate", current_lr, epoch)
                        # save best val model
                        if val_epoch_loss < min_val_loss:
                            min_val_loss = val_epoch_loss
                            # torch.save(model.state_dict(), model_fromVal)
                            print('Best model in val from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromVal))
                # end for batch_idx
            writer.add_scalar('Loss/Val', val_epoch_loss, epoch)
            writer.add_scalar('CI/Val', val_CI, epoch)
            writer.add_scalar('rm2/Val', val_rm2, epoch)
            
            val_mse.append(val_epoch_loss)
            
            
            # print('val_MSE:', '{:.3f}'.format(EM.get_MSE(val_obs, val_pred)))
            # print('val_CI:', '{:.3f}'.format(EM.get_cindex(val_obs, val_pred)))
            # print('val_rm2:', '{:.3f}'.format(EM.get_rm2(val_obs, val_pred)))
            '''
            # record time for 1 epoch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print('Time taken for 1 epoch is {:.4f} minutes'.format((time.time()-start)/60))
            print('\n')
            
            '''
            # 如果验证集准确率提升，更新最佳准确率和重置计数器
            if epoch>100 and val_CI > best_val_CI:
                ep=epoch
                best_val_CI = val_CI
                print('Best model_CI in val from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromValCI))
                # torch.save(model.state_dict(), model_fromValCI)
                counter = 0
            else:
                counter += 1
            # 如果100epoch后连续patience个周期验证集准确率没有提升，触发早停
            # if epoch>100 and counter >= patience:
            if epoch>=400 and counter >= patience:
                print(f"Early stopping triggered. Best Validation CI: {best_val_CI:.6f}")
                print("epoch={}".format(ep))
                break
        # end for epoch
        # np.savetxt('mse.csv', np.array(val_mse), delimiter=',')
        # np.savetxt('CI.csv', np.array(val_ci), delimiter=',')
        # np.savetxt('RM2.csv', np.array(val_RM2), delimiter=',')
        # np.savetxt('trainLossMean_list.csv', trainEP_loss_list, delimiter=',')
        '''
        
        
    
    
    
    