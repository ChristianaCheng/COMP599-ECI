#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:40:48 2022

@author: teishiryou
"""

import random
import stanza
import networkx as nx

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

label_file ='/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TempEval3-eval.txt'
document_dir = '/Users/teishiryou/desktop/research/causality/data/CausalTM//TempEval3-eval_COL/'

tempeval_results = generate_data(label_file, document_dir)

label_file = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank.CLINK.txt'
document_dir = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank_COL/'
causalTB_results = generate_data(label_file, document_dir)


random.shuffle(causalTB_results)
l = int(len(causalTB_results) / 10 * 9)

train_set = causalTB_results[:l]
test_set = causalTB_results[l:]    

#print(len(train_set))

try_data = []

for i in range(10):
    data = train_set[i]
    try_data.append(data)

#print(len(test_set))
#print(try_data)

#data = ['wsj_0610.col', \
        #['General', 'Mills', ',', 'meanwhile', ',', 'finds', 'itself', 'constrained', 'from', 'boosting', 'sales', 'further', 'because', 'its', 'plants', 'are', 'operating', 'at', 'capacity', '.'], \
        #[7], [16], 'Cause']
data = try_data[0]
def make_nlp():
    nlp = stanza.Pipeline('en',processors='tokenize,mwt,pos,lemma,depparse',tokenize_pretokenized=True)
    return nlp

#nlp = make_nlp()
#text = [['A', 'major', 'goal', 'of', 'Kuchma', "'s", 'four-day', 'state', 'visit', 'was', 'the', 'signing', 'of', 'a', '10-year', 'economic', 'program', 'aimed', 'at', 'doubling', 'the', 'two', 'nations', "'", 'trade', 'turnover', ',', 'which', 'fell', 'to', 'dlrs', '14', 'billion', 'last', 'year', ',', 'down', 'dlrs', '2.5', 'billion', 'from', '1996', '.']]

def generate_text_entities_and_label(data):
    
    sentence = [data[1]]
    i = data[2][0]
    j = data[3][0]

    name_i = sentence[0][i]
    name_j = sentence[0][j]
    label = data[4]
    
    entity1 = '{0}-{1}'.format(name_i,i)
    entity2 = '{0}-{1}'.format(name_j,j)
    return entity1,entity2,label,sentence

#entity1,entity2,label,text = generate_text_entities_and_label(data)
#print(entity1,entity2,label,text)

def make_doc(nlp,text):
    return nlp(text)

#doc = make_doc(nlp,text)

def make_dict(doc):
    return doc.sentences[0].to_dict()
# Convert sentence object to dictionary  

#sent_dict = make_dict(doc)


def visualization(sent_dict):
    print ("{:<15} | {:<10} | {:<15} ".format('Token', 'Relation', 'Head'))
    print ("-" * 50)
    for word in sent_dict:
        #if word['head'] == 0:
         #   root=word['text']
         #   root_id=word['id']
        print ("{:<15} | {:<10} | {:<15} "
               .format(str(word['text']),str(word['deprel']), str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))

#visualization(sent_dict)

def find_root_and_id(sent_dict):
    # iterate to print the token, relation and head
    root,root_id = None, None
    for word in sent_dict:
        if word['head'] == 0:
            root=word['text']
            root_id=word['id']-1
    return root,root_id

#root,root_id = find_root_and_id(sent_dict)
#print(root,root_id)

def find_shortest_path(doc,entity1,entity2):
    edges = []
    for token in doc.sentences[0].dependencies:
        if token[0].text.lower() != 'root':
            #edges.append((token[0].text.lower(), token[2].text))
            starting = '{0}-{1}'.format(token[0].text.lower(),int(token[0].id-1))
            ending = '{0}-{1}'.format(token[2].text.lower(),int(token[2].id-1))
            edges.append((starting,ending))
    graph = nx.Graph(edges)   
    SDP = nx.shortest_path(graph, source=entity1, target=entity2)
    return SDP

#SDP = find_shortest_path(doc,entity1,entity2)
#print(SDP)

def find_dep_relations_and_pos(SDP, sent_dict):
    dep_relations = []
    pos_tags = []
    for node in SDP:
        text, index = node.split('-')
        index = int(index)

        for word in sent_dict:
            if (word['id']-1) == index:
                dep_relations.append(word['deprel'])
                pos_tags.append(word['xpos'])
                break
    return dep_relations,pos_tags

#dep_relations,pos_tags = find_dep_relations_and_pos(SDP,sent_dic)

#print(dep_relations,pos_tags)

def find_root_index_in_SDP(SDP,root_id):
    index = 0
    root_index = -1
    
    for node in SDP:
        text, i = node.split('-')
        i = int(i)
        #print(i)
        if i == root_id:
            root_index = index
        index += 1
    return root_index


#root_index = find_root_index_in_SDP(SDP,root_id)
#print(root_index)



def generate_two_subpaths(SDP,root_index,dep_relations,pos_tags):

    n = len(SDP)
    subpath1= []
    subpath2 = []
    
    sub_SDP1 = []
    sub_SDP2 = []
    sub_dep1 = []
    sub_dep2 = []
    sub_pos1 =[]
    sub_pos2 = []
    
    for i in range(root_index+1):
        text, index = SDP[i].split('-')
        index = int(index)
        sub_SDP1.append(text)
        sub_dep1.append(dep_relations[i])
        sub_pos1.append(pos_tags[i])
    subpath1.append(sub_SDP1)
    subpath1.append(sub_dep1)
    subpath1.append(sub_pos1)
        
    for j in range(n-1,root_index-1,-1):
        text, index = SDP[j].split('-')
        
        index = int(index)
        sub_SDP2.append(text)
        sub_dep2.append(dep_relations[j])
        sub_pos2.append(pos_tags[j])
    subpath2.append(sub_SDP2)
    subpath2.append(sub_dep2)
    subpath2.append(sub_pos2)

    return subpath1,subpath2

'''
def get_seperate_input(data):
    nlp = make_nlp()
    entity1,entity2,label,text = generate_text_entities_and_label(data)
    doc = make_doc(nlp,text)
    sent_dict = make_dict(doc)
    visualization(sent_dict)
    root,root_id = find_root_and_id(sent_dict)
    SDP = find_shortest_path(doc,entity1,entity2)
    dep_relations,pos_tags = find_dep_relations_and_pos(SDP,sent_dict)
    root_index = find_root_index_in_SDP(SDP,root_id)
    
    if root_index != -1:
        print('root in SDP')
        subpath1 ,subpath2 = generate_two_subpaths(SDP,root_index,dep_relations,pos_tags)  
        print(subpath1)
        print('\n')
        print(subpath2)
        
    else:
        print('no root in SDP')
        print(SDP)
        
for data in try_data:
    get_seperate_input(data)              
'''
nlp = make_nlp()
entity1,entity2,label,text = generate_text_entities_and_label(data)
print(entity1,entity2,text)
doc = make_doc(nlp,text)
sent_dict = make_dict(doc)
root,root_id = find_root_and_id(sent_dict)
SDP = find_shortest_path(doc,entity1,entity2)
visualization(sent_dict)
print(SDP)
dep_relations,pos_tags = find_dep_relations_and_pos(SDP,sent_dict)
root_index = find_root_index_in_SDP(SDP,root_id)

print(root_index)
print(root)
if root_index != -1:
    print('root in SDP')
    subpath1 ,subpath2 = generate_two_subpaths(SDP,root_index,dep_relations,pos_tags)  
    print(subpath1)
    print('\n')
    print(subpath2)
        
        