#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:24:08 2022

@author: teishiryou
"""
import gensim
from gensim.models import Word2Vec
import gensim.downloader
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np

#love_vectors = gensim.downloader.load('glove-wiki-gigaword-200')


#model = Word2Vec(sent, size = 200, sg = 1)
#model.save("word2vec.model")

# convert glove file to word2vec file first in the terminal
#python -m gensim.scripts.glove2word2vec --input code/glove.6B.200d.txt --output data/word2vec_from_glove_200.txt
#https://radimrehurek.com/gensim/scripts/glove2word2vec.html

# Transfer learning” on Google pre-trained word2vec
# https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights
sent = [['helped'], ['continue', 'allowed', 'helped'], ['facing', 'inc.', 'said'], ['squeeze', 'facing', 'inc.', 'said'], ['said'], ['roundup', 'was', 'said'], ['launch', 'be'], ['improve', 'seize', 'hoping', 'launch', 'be'], ['see', 'surprised', 'said'], ['represent', 'surprised', 'said'], ['started'], ['for', 'sale', 'said', 'started'], ['declined'], ['begin', 'said', 'declined'], ['continue', 'said'], ['pray', 'continue', 'said'], ['said'], ['investigate', 'said'], ['quarantine', 'hopes'], ['staunch', 'hopes']]
# handling out-of-vocab
# https://stackoverflow.com/questions/57662405/is-it-appropriate-to-train-w2v-model-on-entire-corpus
sent.append(['unk'])
model = Word2Vec(sent, vector_size =200, sg = 1,min_count=1)

model.build_vocab(sent)
model.wv.vectors_lockf = np.ones(len(model.wv))
total_examples = model.corpus_count
model_base = KeyedVectors.load_word2vec_format('word2vec_from_glove_200.txt',binary=False)
print(model_base['computer'])
#https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
model.build_vocab([list(model.wv.key_to_index)], update=True)
model.wv.vectors_lockf = np.ones(len(model.wv))
#https://stackoverflow.com/questions/69412142/process-to-intersect-with-pre-trained-word-vectors-with-gensim-4-0-0
model.wv.intersect_word2vec_format("word2vec_from_glove_200.txt", binary=False, lockf=1.0)
model.train(sent, total_examples=total_examples, epochs=model.epochs)

# store the full model that can be trained further
#model.save("word2vec.model")
# The trained word vectors are stored in a KeyedVectors instance, as model.wv
# vector = model.wv['computer']  # get numpy vector of a word

# store only the keyvalue pairs
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
print(wv)

words = ['computer','helped']
# https://stackoverflow.com/questions/30301922/how-to-check-if-a-key-exists-in-a-word2vec-trained-model-or-not
for word in words:
    if word in wv.key_to_index:
        print(wv[word])
    else:
        print('out-of-vocab')
        print(wv['unk'])
        
# vector = wv['computer']
# if we finish training del model