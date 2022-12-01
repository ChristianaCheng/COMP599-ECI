# -*- coding: utf-8 -*-


import pickle
import random
import json
import requests
import numpy as np
import pickle
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel

"""## Search Definitions in ConceptNet

In here we only consider 18 semantic relations.
"""

def crawl_concept_net(event):
    obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
    relations = ['CapableOf', 'IsA', 'HasProperty', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent', 'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor', 'CreatedBy', 'MadeOf', 'Desires']
    res = []
    for e in obj['edges']:
        if e['rel']['label'] in relations:
            res.append(' '.join([e['rel']['label'], e['end']['label']]))
        
    return res

"""## Load data from json

data = {'intra':[x,x,x...]}
"""

def load_data(mode, filename):
  data = {mode: [{line} for line in open(filename, 'r')]}
  return data
data = load_data('intra', "EventStoryLine9.json")

"""## Knowledge Linearization"""

def linear_knowledge(element):
  sent1,sent2,event1,event2 = element['sentence1'],element['sentence2'],element['event1'],element['event2']
  know_e1 = crawl_concept_net(event1)[:5]
  know_e2 = crawl_concept_net(event2)[:5]
  linear_e1,linear_e2 = [], []
  for i in know_e1:
    count = 0
    ls = i.split()
    for j in ls:
      if count == 0:
        event_mark = '<' + ls[0] + '>'
        linear_e1.append(event_mark)
      else:
        linear_e1.append(j)
      count += 1
  for i in know_e1:
    count = 0
    ls = i.split()
    for j in ls:
      if count == 0:
        event_mark = '<' + ls[0] + '>'
        linear_e2.append(event_mark)
      else:
        linear_e2.append(j)
      count += 1
  return linear_e1,linear_e2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def add_knowledge(element):
  sent1,sent2,event1,event2 = element['sentence1'],element['sentence2'],element['event1'],element['event2']
  linear_e1,linear_e2 = linear_knowledge(element)
  mark_e1,mark_e2 = ['<e1>'],['<e2>'] 
  for i in linear_e1:
    mark_e1.append(i)
  for j in linear_e2:
    mark_e2.append(j)
  mark_e1.append('</e1>')
  mark_e2.append('</e2>')

  know_s1,know_s2 = [], []
  for w in sent1:
    if w == event1:
      for e1 in mark_e1:
        know_s1.append(e1)
    else:
      know_s1.append(w)

  for w in sent2:
    if w == event2:
      for e2 in mark_e2:
        know_s2.append(e2)
    else:
      know_s2.append(w)
  return know_s1,know_s2

def add_token(element,mask = True):
  dataset = []
  sent1,sent2,event1,event2 = element['sentence1'],element['sentence2'],element['event1'],element['event2']
  start1 = 0
  span1_vec = []
  span2_vec = [] 

  temp = []
  t = ''
  for i in sent1:
    if i == event1:
      break
    else:
      temp.append(i)
  temp_len = len(''.join(temp))

  temp = []
  t = ''
  for i in sent2:
    if i == event2:
      break
    else:
      temp.append(i)
  temp_len = len(''.join(temp))

  linear1,linear2 = linear_knowledge(element)
  span1=''.join(linear1)
  span2=''.join(linear2)

  span1_len = len(span1)
  c1 = 0
  while c1 < span1_len:
    span1_vec.append(c1+temp_len)
    c1 += 1

  span2_len = len(span2)
  c2 = 0
  while c2 < span2_len:
    span2_vec.append(c2+temp_len)
    c2 += 1

  sent1, sent2 = add_knowledge(element)
  token_s1 = ['[CLS]'] + sent1 + ['[SEP]']
  token_s2 = ['[CLS]'] + sent2 + ['[SEP]']
  s1, s2 = ''.join(token_s1), ''.join(token_s2)
  sentence_vec_s = []
  sentence_vec_t = []

  

  # for i, w in enumerate(span1):
  #     tokens = tokenizer.tokenize(w)
  #     xx = tokenizer.convert_tokens_to_ids(tokens)
  #     span1_vec.extend(xx)
  # for i, w in enumerate(span2):
  #     tokens = tokenizer.tokenize(w)
  #     xx = tokenizer.convert_tokens_to_ids(tokens)
  #     span2_vec.extend(xx)

  for i, w in enumerate(s1):
      tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
      xx = tokenizer.convert_tokens_to_ids(tokens)
      sentence_vec_s.extend(xx)

  for i, w in enumerate(s2):
      tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
      xx = tokenizer.convert_tokens_to_ids(tokens)
      sentence_vec_t.extend(xx)

  

  if mask:
    for i in span1_vec:
      
      sentence_vec_s[i] = 103
    for j in span2_vec:
      sentence_vec_t[i] = 103

  dataset.append(sentence_vec_s)
  dataset.append(sentence_vec_t)
  
  return dataset

def main():
  dataset = []
  for e in data:
    sent1,sent2,event1,event2 = e['sentence1'],e['sentence2'],e['event1'],e['event2']
    sent = add_token(e)
    doc = e['document']
    label = 0 if e['relation'] == 'NULL' else 1
  dataset.append([doc,sent1, sent2, event1,event2, sent, label])
  return dataset

"""## Eval"""

def evaluate(ref,pred):
  c_predict = 0
  c_correct = 0
  c_gold = 0

  for g, p in zip(ref, pred):
      if g != 0:
           c_gold += 1
      if p != 0:
           c_predict += 1
      if g != 0 and p != 0:
          c_correct += 1

  p = c_correct / (c_predict + 1e-100)
  r = c_correct / c_gold
  f = 2 * p * r / (p + r + 1e-100)

  print('correct', c_correct, 'predicted', c_predict, 'golden', c_gold)
  return p, r,

"""## BERT Encoder"""

class BertModel(nn.Module):
    def __init__(self, y_num):
        super(BertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(768 * 2, y_num)
    
    def forward(self, sentences_1, mask_s, sentences_2, mask_t, event1, event2):
        if self.training:
            self.bert.train()
            encoded_layer_s,_ = self.bert(sentences_1, mask_s, output_all_encoded_layers=False)
            encoded_layer_t,_ = self.bert(sentences_2, mask_t, output_all_encoded_layers=False)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layer_s,_ = self.bert(sentences_1, mask_s, output_all_encoded_layers=False)
                encoded_layer_t,_ = self.bert(sentences_2, mask_t, output_all_encoded_layers=False)
        
        event1 = torch.cat([encoded_layer_s[i][event1[i][0]:event1[i][1]] for i in range(len(event1))], dim=0)
        event2 = torch.cat([encoded_layer_t[i][event2[i][0]:event2[i][1]] for i in range(len(event2))], dim=0)
        
        opt1 = torch.sum(event1, dim=0)
        opt2 = torch.sum(event2, dim=0)
        opt = torch.cat([opt1, opt2], dim=0)
        opt = self.drop(opt)
        opt = self.fc(opt)
        return opt

