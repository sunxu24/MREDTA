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

d_model_d = int(768) # Embedding Size for drug
d_model_t = int(1024) # Embedding Size for target
d_ff = 512 # FeedForward dimension
d_k = d_v = 16 # dimension of K(=Q), V
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

class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb):
        self.texta = texta
        self.textb = textb
        # self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item]

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
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
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
    def forward(self, enc_inputs, tokenizer, BERT_model, batch_idx, key, process):
        # key: Drug or Target
        # process: train or val
        bert_inputs_path = "./BERT_data_iso/BERT_data_64/BERT_EGFR_case1"  + "/BERT_inputs/Batch" + str(batch_idx) + ".pt" 
        bert_inputs = torch.load(bert_inputs_path)   
        bert_results_path = "./BERT_data_iso/BERT_data_64/BERT_EGFR_case1" + "/BERT_results/Batch" + str(batch_idx) + ".pt" 
        bert_results = torch.load(bert_results_path)   
        bert_attentions_path = "./BERT_data_iso/BERT_data_64/BERT_EGFR_case1" + "/BERT_attentions/Batch" + str(batch_idx) + ".pt" 
        bert_attentions = torch.load(bert_attentions_path)   
        
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

    def forward(self, input_Drugs, input_Tars, batch_idx, process):
        # print("#### Transformer ####")
        # input: [batch_size, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # print(type(input_Drugs[0]))
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs,self.tokenizer_drug,self.model_drug, batch_idx, "Drug", process)
        # print(f"enc_Drugs.size()={enc_Drugs.size()}")
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars, batch_idx, process)
        # print(f"enc_Tars.size()={enc_Tars.size()}")
        
        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        # print(f"enc_Drugs_2D1:{enc_Drugs_2D1.shape}")
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        # print(f"enc_Tars_2D1:{enc_Tars_2D1.shape}")
        #fc = enc_Drugs_2D1 + enc_Tars_2D1
        fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)

        fc0 = self.fc0(fc)
        # fc1 = self.fc1(fc0)
        # fc2 = self.fc2(fc1)
        # 去掉fc1层
        fc2 = self.fc2(fc0)
        affi = fc2.squeeze()

        return affi, enc_attnsD0, enc_attnsT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2

if __name__ == '__main__':
    ############# Test Process ############
    
    # load_model = "./backup_kiba/no_fc1/backup_no_fc1_lr4/models/model_fromValCI.pth"
    load_model = "./backup_kiba/batchsize64/allResults_newKIBA_1215/model_mse176_ci875_rm710.pth"
    # load_model = "./backup_kiba/batchsize64/KIBAiso_ep600_allResults23-12-24/models/model_fromVal_CI880_mse146_rm2720.pth"
    
      
    if torch.cuda.is_available():
        model = Transformer().cuda()
    else:
        model = Transformer().cpu()
        
    pretrained_weights = torch.load(load_model) # 加载预训练模型
    model.load_state_dict(pretrained_weights)
    
    model.eval()
    
    criterion = torch.nn.MSELoss(reduction='mean')
    
    # load data
    fpath_EGFR = './data_EGFR_case1'
    # fpath_EGFR = './newEGFR'
    drug, target, affinity = DH.LoadData(fpath_EGFR, logspance_trans=False)
    print(f"len(drug)={len(drug)}")
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('EGFR', drug, target, affinity)
    print(f"len(drug_seqs)={len(drug_seqs)}")
    # print(len(drug_seqs))
    # len(drug_seqs)=2091, len(target_Seqs)=2091
    batch_size = 64
    test_iter = DatasetIterater(drug_seqs, target_seqs)
    # test_loader = Data.DataLoader(test_iter, batch_size, False)
    test_loader = Data.DataLoader(test_iter, batch_size, drop_last=True)
    
    seed_torch(seed=2)
    process = "val"
 
    test_pred=[]
    affi_pre=[]
    
    model.eval()
    with torch.no_grad():
        for test_batch_idx, (SeqDrug, SeqTar) in enumerate(test_loader):
            # print(f"第{test_batch_idx} batch 开始！！！")
            # real_affi = real_affi.cuda()
            pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2 \
                                                                        = model(SeqDrug, SeqTar, test_batch_idx, process)
            # affi_pre.append(pre_affi)
            
            test_pred.extend(pre_affi.tolist())

        np.savetxt('pre_affi_iso_drug1_prot2.csv', np.array(test_pred), delimiter=',')