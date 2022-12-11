import pickle
import random
import numpy as np
import json
import argparse
import requests
import spacy
from conceptnet_utils import wordify_knowledge,crawl_concept_net,linear_knowledge

nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")

#---------------------------------------------------------------
def get_sentence_number(s, all_token):
    '''
    To get the sentence id based on a word id
    input:
        s(int): word id
        all_token: the article with each word represents as a tuple of
                    (word_id, sentence_id, word_id_in_the_sentence, word)
                    Note: word_id begins at 1, 
                    sentence_id begins at 0, 
                    token id in sentence begins at 0
    output:
        the id of the sentence which includes the word with the unique id
    '''
    token_id = s.split('_')[0] #token_id
    for token in all_token:
        if token[0] == token_id:
            sent_id = token[1] 
            return sent_id

def nth_sentence(sen_no):
    '''
    To get the tokens of sentence n
    input:
        sen_no(int): the sentence id
    output:
        the sentence with the specified id
    ''' 
    res = []
    for token in all_token:
        if token[1] == sen_no:
            res.append(token[-1])
    return res

def get_sentence_offset(s, all_token):
    '''
    input: 
        s: token id(s) for the event
        all_token: article
    output:
        token ids in the sentence representing the event
    '''
    positions = []
    for c in s.split('_'):
        token = all_token[int(c) - 1]
        positions.append(token[2])
    return '_'.join(positions)

def get_token(token_id):
    '''
    input:
        token_id: token id(s) for the event
    output:
        token(s) for the event
    '''
    token_ids = token_id.split('_')
    tokens = []
    for token_id in token_ids:
        for token in all_token:
            if token[0] == token_id:
                 tokens.append(token[3])
                 break
    return ' '.join(tokens)
#------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conceptNet',type=int,default=1)
    parser.add_argument('--version',type=str,default='v0.9')
    args = parser.parse_args()
    version = args.version # version can be 0.9 and 1.0
    knowledge = args.conceptNet # whether or not to include external information from conceptNet
    # baseline bert: set knowledge = 0
    # bert with external knowledge: set knowledge = 1


    with open(f'document_raw_{version}.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            documents = pickle.load(f)
    counter = 0
    intra = 0
    inter = 0
    add = ''
    if knowledge:
        add = "_conceptNet"
    fname_val = f'EventStoryLine_test_{version}{add}.json'
    fname_train = f'EventStoryLine_train_{version}{add}.json'
    val = open(fname_val,'w',encoding ='utf-8')
    train = open(fname_train,'w',encoding ='utf-8')
    relations = set()
    val_topics = ["37", "41"]
    for doc in documents:
        valid_pairs = []
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
        evaluation_data, evaluationcrof_data] = documents[doc]

        topic = doc.split(version+'/')[1].split('/')[0]

        events = list(ecb_star_events.values())
        for i, offset1 in enumerate(events):
            # this is to avoid duplicate tuple of events
            # since we only consider if e1,e2 has causal relations
            # but not e1 -> e2 and e2 -> e1
            for j,offset2 in enumerate(events[i+1:]):
                if offset1 == offset2:
                    continue

                ##Causal Relation
                rel = 'NULL'
                # search in the causal set to see if the relation e1 -> e2 has causal relation
                for elem in evaluation_data:
                    e1, e2, value = elem
                    if e1 == offset1 and e2 == offset2:
                        rel = value
                        relations.add(rel)
                        break
                # if e1 -> e2 does not have causal relation
                # search if e2 -> e1 has causal relation
                if rel == "NULL":
                    for elem in evaluation_data:
                        e1, e2, value = elem
                        if e2 == offset1 and e1 == offset2:
                            rel = value
                            relations.add(rel)
                            break
                #this way we only have one tuple of either (e1,e2) or (e2,e1) in our corpus
                
                sen_s = get_sentence_number(offset1, all_token)
                sen_t = get_sentence_number(offset2, all_token)
                e1 = get_token(offset1)
                e2 = get_token(offset2)
                sentence_s = nth_sentence(sen_s)
                sentence_t = nth_sentence(sen_t)
                sen_offset1 = get_sentence_offset(offset1, all_token)
                sen_offset2 = get_sentence_offset(offset2, all_token)
                # mapping labels
                # possible labels: null, precondition, falling action
                if rel == 'NULL' or rel == 'null':
                    label = 0
                elif rel == 'FALLING_ACTION' or rel == 'PRECONDITION':
                    label = 1
                data = {
                    'event1_id':sen_offset1.split("_"),
                    'event2_id':sen_offset2.split("_"),
                    'event1':e1,
                    'event2':e2,
                    'sentence1_id':sen_s,
                    'sentence2_id':sen_t,
                    'sentence1':sentence_s,
                    'sentence2':sentence_t,
                    'relation':rel,
                    'label':label,
                    'topic':topic,
                    'ducument':doc,
                    #'article':all_token # if we want to add the whole article the json file
                }
                e1_know,e2_know = linear_knowledge(data)
                add1,add2 ="",""
                if knowledge:
                    add1,add2 = e1_know,e2_know
                
                # add event markers to the sentence
                if data['sentence1_id'] == data['sentence2_id']:
                  intra += 1
                  if rel != 'NULL' and rel != 'null':
                    counter += 1
                  temp = data["sentence1"]
                  sentence = temp.copy()
                  e1_id,e2_id = data['event1_id'],data['event2_id']
                  e1_id,e2_id = [int(i) for i in e1_id],[int(i) for i in e2_id]
                  
                  
                  if e1_id[0] > e2_id[0]:
                    
                    sentence.insert(e1_id[0],"<e1>")
                    sentence.insert(e1_id[-1]+2,"</e1>")
                    sentence.insert(e1_id[0]+2,add1)
                    sentence.insert(e2_id[0],"<e2>")
                    sentence.insert(e2_id[-1]+2,"</e2>")
                    sentence.insert(e2_id[0]+2,add2)
                  elif e1_id[0] <= e2_id[0]:
                    sentence.insert(e2_id[0],"<e2>")
                    sentence.insert(e2_id[-1]+2,"</e2>")
                    sentence.insert(e2_id[0]+2,add2)
                    sentence.insert(e1_id[0],"<e1>")
                    sentence.insert(e1_id[-1]+2,"</e1>")
                    sentence.insert(e1_id[0]+2,add1)
  
                  sentence = " ".join(sentence)
                  data.update({'marked_sent1':sentence,
                    'marked_sent2':"",
                    'intra':'true',
                    'sentence1':" ".join(data['sentence1']),
                    'sentence2':" ".join(data['sentence2'])
                    })
         
                  if topic in val_topics:
                        json.dump(data,val,ensure_ascii=False)
                        val.write('\n')

                  else:
                        json.dump(data,train,ensure_ascii=False)
                        train.write('\n')
                else:
                    inter += 1
                    if rel != 'NULL' and rel != 'null':
                        counter += 1
                    temp1,temp2 = data["sentence1"],data["sentence2"]
                    sentence1,sentence2 = temp1.copy(),temp2.copy()

                    e1_id,e2_id = data['event1_id'],data['event2_id']
                    e1_id,e2_id = [int(i) for i in e1_id],[int(i) for i in e2_id]

                    sentence1.insert(e1_id[0],"<e1>")
                    sentence1.insert(e1_id[-1]+2,"</e1>")
                    sentence1.insert(e1_id[0]+2,add1)
                    sentence2.insert(e2_id[0],"<e2>")
                    sentence2.insert(e2_id[-1]+2,"</e2>")
                    sentence2.insert(e2_id[0]+2,add2)

                    sentence1 = " ".join(sentence1)
                    sentence2 = " ".join(sentence2)
                    if data["sentence1_id"] > data["sentence2_id"]:
                        temp = sentence1
                        sentence1 = sentence2
                        sentence2 = temp
                        
                    data.update({'marked_sent1':sentence1,
                    'marked_sent2':sentence2,
                    'intra':'false',
                    'sentence1':" ".join(data['sentence1']),
                    'sentence2':" ".join(data['sentence2'])}
                    )


                    if topic in val_topics:
                        json.dump(data,val,ensure_ascii=False)
                        val.write('\n')

                    else:
                        json.dump(data,train,ensure_ascii=False)
                        train.write('\n')


        

    print('Test size: ',sum(1 for line in open(fname_val,'r')))
    print('Train size: ',sum(1 for line in open(fname_train,'r')))
    print("Causal Relations:",counter)
    print('intra:',intra)
    print('inter:',inter)
    val.close()
    train.close()
    
