#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:24:59 2022

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
#print(tempeval_results[9])

label_file = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank.CLINK.txt'
document_dir = '/Users/teishiryou/desktop/research/causality/data/CausalTM/Causal-TimeBank_COL/'
causalTB_results = generate_data(label_file, document_dir)

random.shuffle(causalTB_results)
l = int(len(causalTB_results) / 10 * 9)

train_set = causalTB_results[:l]
test_set = causalTB_results[l:]    

print(len(train_set))
try_data = []

for i in range(10):
    try_data.append(data)
print(len(test_set))

print(try_data)

nlp = stanza.Pipeline('en',processors='tokenize,mwt,pos,lemma,depparse',tokenize_pretokenized=True)
'''
text = [['Oil', 'giant', 'BP', 'has', 'said', 'it', 'will',
         'buy', 'back', '$', '8bn', 'of', 'shares', ',', 
         'returning', 'to', 'shareholders', 'the', 'money', 
         'they', 'had', 'put', 'into', 'a', 'complicated', 
         'Russian', 'venture', '.']]

text=[['It', 'said', 'it', '"', 'expected', 'to', 'return', 'to', 'BP',
       'shareholders', 'an', 'amount', 'equivalent', 'to', 'the', 'value',
       'of', 'the', 'company', "'s", 'original', 'investment', 'in', 
       'TNK-BP', '"', '.']]

text=[['Robots', 'in', 'popular','culture', 'are', 'there', 'to', 'remind',
       'us', 'of', 'the', 'awesomeness', 'of', 'unbound','human' ,
       'agency','.']]

text = [['The','company','said','it','has','agreed','to','sell','the','extrusion','division']]

text = [['A','trillion','gallons','of','water','have','been','poured',
         'into','an','empty','region','of','outer','space','.'  
    ]]
'''

#text = [['A', 'major', 'goal', 'of', 'Kuchma', "'s", 'four-day', 'state', 'visit', 'was', 'the', 'signing', 'of', 'a', '10-year', 'economic', 'program', 'aimed', 'at', 'doubling', 'the', 'two', 'nations', "'", 'trade', 'turnover', ',', 'which', 'fell', 'to', 'dlrs', '14', 'billion', 'last', 'year', ',', 'down', 'dlrs', '2.5', 'billion', 'from', '1996', '.']]
doc = nlp(text)

# doc.sentences[0] is a list of dict of the words with attributes id,text,xpos,head
#print(doc.sentences)
# output: (current_token, head_index, dep_relation)
print(doc.sentences[0].print_dependencies())

#Better visualize the dependency
print ("{:<15} | {:<10} | {:<15} ".format('Token', 'Relation', 'Head'))
print ("-" * 50)

# Convert sentence object to dictionary  
sent_dict = doc.sentences[0].to_dict()
print(sent_dict)
# iterate to print the token, relation and head
for word in sent_dict:
    if word['head'] == 0:
        root=word['text']
        root_id=word['id']
    print ("{:<15} | {:<10} | {:<15} "
           .format(str(word['text']),str(word['deprel']), str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))

print('ROOT IS ',root,'INDEX: ',root_id)

# find shortest dependency path
# Load stanfordnlp's dependency tree into a networkx graph
# version 1
edges = []
for token in doc.sentences[0].dependencies:
    #print(token)
    #print(token[0])
    #print(token[2])
    #print('\n')
    if token[0].text.lower() != 'root':
        #edges.append((token[0].text.lower(), token[2].text))
        starting = '{0}-{1}'.format(token[0].text.lower(),token[0].id)
        ending = '{0}-{1}'.format(token[2].text.lower(),token[2].id)
        edges.append((starting,ending))
print(edges)
graph = nx.Graph(edges)

'''
edges = []
dependencies = {}

for edge in doc.sentences[0].dependencies:
    edges.append((edge['governor'], edge['dependent']))
    dependencies[(min(edge['governor'], edge['dependent']),
                  max(edge['governor'], edge['dependent']))] = edge
    
graph = nx.Graph(edges)
'''
# Get the length and path
# the id is starting from 1
entity1 = 'visit-9'
entity2 = 'doubling-20'
#entity1 = text[0][4]
#entity2 = text[0][11]
print(nx.shortest_path_length(graph, source=entity1, target=entity2))
print(nx.shortest_path(graph, source=entity1, target=entity2))

SDP = nx.shortest_path(graph, source=entity1, target=entity2)

#print(SDP)
def find_dep_relations_and_pos(SDP):
    dep_relations = []
    pos_tags = []
    for node in SDP:
        text, index = node.split('-')
        index = int(index)

        for word in sent_dict:
            if word['id'] == index:
                dep_relations.append(word['deprel'])
                pos_tags.append(word['xpos'])
                break
    return dep_relations,pos_tags

dep_relations,pos_tags = find_dep_relations_and_pos(SDP)

print(dep_relations,pos_tags)
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


root_index = find_root_index_in_SDP(SDP,root_id)

print(root_index)



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

if root_index != -1:
    subpath1 ,subpath2 = generate_two_subpaths(SDP,root_index,dep_relations,pos_tags)  
    print(subpath1)
    print('\n')
    print(subpath2)


