#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:06:14 2022

@author: teishiryou
"""
'''
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = ['While', 'partisan', 'division', 'is', 'the', 'central', 
        'characteristic', 'of', 'the', 'modern', 'Congress', ',', 
        'women', 'have', 'begun', 'to', 'crack', 'away', 'at', 
        'the', 'gridlock', 'by', 'forming', 'coalitions', 
        'that', 'have', 'surprised', 'leaders', 'of', 'both', 
        'parties', '.']
#text = " ".join(text)


result = nlp.annotate(text,
                   properties={
                       'annotators': 'tokenize, ssplit, pos',
                       'outputFormat': 'json',
                       'timeout': 1000,
                       
                   })

pos = []
for word in result["sentences"][0]["tokens"]:
    pos.append('{} ({})'.format(word["word"], word["pos"]))
    
print(" ".join(pos))
'''

import stanza
#stanza.download('en')

nlp = stanza.Pipeline('en',processors='tokenize,pos',tokenize_pretokenized=True)
text = [['Oil', 'giant', 'BP', 'has', 'said', 'it', 'will',
         'buy', 'back', '$', '8bn', 'of', 'shares', ',', 
         'returning', 'to', 'shareholders', 'the', 'money', 
         'they', 'had', 'put', 'into', 'a', 'complicated', 
         'Russian', 'venture', '.']]

doc = nlp(text)
sent_list = [sent.text for sent in doc.sentences]
print([f'{word.text} {word.xpos}' for sent in doc.sentences for word in sent.words])
# [4], [7]
# ['Oil NN', 'giant NN', 'BP NNP', 'has VBZ', 'said VBN', 
#'it PRP', 'will MD', 'buy VB', 'back RB', '$ $', '8bn CD', 
#'of IN', 'shares NNS', ', ,', 'returning VBG', 'to IN', 
#'shareholders NNS', 'the DT', 'money NN', 'they PRP', 
#'had VBD', 'put VBN', 'into IN', 'a DT', 'complicated JJ', 
#'Russian JJ', 'venture NN', '. .']
    
    
    
    
    
    
    
    
    
    
    