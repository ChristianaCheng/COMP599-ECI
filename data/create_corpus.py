import pickle
import random
import numpy as np
import json
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


if __name__ == '__main__':
    version = 'v0.9'
    with open(f'document_raw_{version}.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            documents = pickle.load(f)

    intra = open(f'EventStoryLine_intra_{version}.json','w',encoding ='utf-8')
    inter = open(f'EventStoryLine_inter_{version}.json','w',encoding ='utf-8')
    relations = set()

    for doc in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
        evaluation_data, evaluationcrof_data] = documents[doc]
        topic = doc.split(version+'/')[1].split('/')[0]

        for event1 in ecb_star_events:
            for event2 in ecb_star_events:
                if event1 == event2:
                    continue

                offset1 = ecb_star_events[event1]
                offset2 = ecb_star_events[event2]

                ##Causal Relation
                rel = 'NULL'
                for elem in evaluation_data:
                    e1, e2, value = elem
                    if e1 == offset1 and e2 == offset2:
                        rel = value
                        relations.add(rel)

                if (rel == 'NULL' or rel == 'null') and event1 > event2:
                    # if event_id1 > event_id2 and its relation is null
                    # either they don't have causal relations
                    # either the pair has the inverse causal relations
                    continue
                # TODO: post-process
                # remove the duplicate case where event1 < event2 has null and event2 > event1 has causal

                sen_s = get_sentence_number(offset1, all_token)
                sen_t = get_sentence_number(offset2, all_token)
                e1 = get_token(offset1)
                e2 = get_token(offset2)
                sentence_s = nth_sentence(sen_s)
                sentence_t = nth_sentence(sen_t)
                sen_offset1 = get_sentence_offset(offset1, all_token)
                sen_offset2 = get_sentence_offset(offset2, all_token)

                data = {
                    'event1_id':sen_offset1,
                    'event2_id':sen_offset2,
                    'event1':e1,
                    'event2':e2,
                    'sentence1_id':sen_s,
                    'sentence2_id':sen_t,
                    'sentence1':sentence_s,
                    'sentence2':sentence_t,
                    'relation':rel,
                    'topic':topic,
                    'ducument':doc,
                    'article':all_token
                }

                if abs(int(sen_s)-int(sen_t)) == 0:
                    json.dump(data,intra,ensure_ascii=False)
                    intra.write('\n')

                else:
                    json.dump(data,inter,ensure_ascii=False)
                    inter.write('\n')
        
    print('intra size: ',sum(1 for line in open(f'EventStoryLine_intra_{version}.json','r')))
    print('inter size: ',sum(1 for line in open(f'EventStoryLine_inter_{version}.json','r')))
    print(relations)
    inter.close()
    intra.close()
    
