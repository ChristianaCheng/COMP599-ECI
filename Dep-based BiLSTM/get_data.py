#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:13:29 2022

@author: teishiryou
"""
import random 
import pandas as pd
import numpy as np
import pickle as pkl
random.seed(42)

def get_conll_data_from_file(filename):
    datasets = []
    tokens, events = list(), list()
    for line in open(filename, encoding='utf-8'):
        if line.isspace():
            if len(tokens) > 0:
                datasets.append([tokens, events])
                tokens, events = list(), list()
        else:
            line = line.strip().split('\t')
            word, event = line[0], line[4]

            tokens.append(word)
            events.append(event)

    if len(tokens) > 0:
        datasets.append([tokens, events])
    return datasets


def generate_data(label_file, document_dir):
    causal_dict = {}
    for line in open(label_file):
        fields = line.strip().split('\t')
        doc_name, e1, e2, _ = fields
        causal_dict.setdefault(doc_name, set())
        causal_dict[doc_name].add((e1, e2))
        causal_dict[doc_name].add((e2, e1))


    results = list()
    for doc in causal_dict:
        event_set = causal_dict[doc]
        filein = document_dir + doc[:-3] + 'col'
        #print(filein)
        datasets = []
        try:
            datasets = get_conll_data_from_file(filein)
        except:
            print(doc)

        for data in datasets:
            tokens, events = data
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    e1 = events[i]
                    e2 = events[j]
                    if e1 != 'O' and e2 != 'O':
                        label = 'NULL'
                        if (e1, e2) in event_set:
                            label = 'Cause'
                        results.append([doc, tokens, [i], [j], label])

    return results


if __name__ == '__main__':
    label_file ='/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TempEval3-eval.txt'
    document_dir = '/Users/teishiryou/desktop/research/causality/data/CausalTM//TempEval3-eval_COL/'

    tempeval_results = generate_data(label_file, document_dir)
    #print(tempeval_results[9])
    
    label_file = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank.CLINK.txt'
    document_dir = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank_COL/'
    causalTB_results = generate_data(label_file, document_dir)
   
    '''
    for i in range(len(tempeval_results)):
        if tempeval_results[i][-1] == 'Cause':
            print(tempeval_results[i])
            print('\n')
'''
    
random.shuffle(causalTB_results)
l = int(len(causalTB_results) / 10 * 9)


train_set = []+causalTB_results[:l]
test_set = []+causalTB_results[l:]  

df = pd.DataFrame(train_set)
df.to_csv('train.csv',index = True)
print(isinstance(df, pd.DataFrame))
df = pd.DataFrame(test_set)
df.to_csv('test.csv',index =True)


    