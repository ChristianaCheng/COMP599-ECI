import requests
import spacy

nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")

def crawl_concept_net(event,normalize=True):
    if normalize:
      event = " ".join([token.lemma_ for token in nlp(event)])
    obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
    relations = ['RelatedTo','Causes','CausesDesire','HasPrerequisite']
    #relations = ['CapableOf', 'IsA', 'HasProperty', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent', 'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor', 'CreatedBy', 'MadeOf', 'Desires']
    res = []
    for e in obj['edges']:
        if e['rel']['label'] in relations:
            res.append(' '.join([e['rel']['label'], e['end']['label']]))
        
    return list(set(res))

def linear_knowledge(element):
  sent1,sent2,event1,event2 = element['sentence1'],element['sentence2'],element['event1'],element['event2']
  # take the first 5 knowledge
  know_e1 = crawl_concept_net(event1)[:5]
  know_e2 = crawl_concept_net(event2)[:5]
  linear_e1,linear_e2 = [], []
  linear_e1 = ["<"+t+">" if i == 0 else t+" <s>" if i==len(e.split())-1 else t for e in know_e1 for i,t in enumerate(e.split())]
  linear_e2 = ["<"+t+">" if i == 0 else t+' <s>' if i==len(e.split())-1 else t for e in know_e2 for i,t in enumerate(e.split())]
  return " ".join(linear_e1)," ".join(linear_e2)

def wordify_knowledge(element):
  sent1,sent2,event1,event2 = element['sentence1'],element['sentence2'],element['event1'],element['event2']
  know_e1 = crawl_concept_net(event1)[:5]
  know_e2 = crawl_concept_net(event2)[:5]
  #print(know_e1,know_e2)
  knowledge = []
  for i,k in enumerate([know_e1,know_e2]):
    if i == 0:
      event = event1
    else:
      event = event2
    relatedTo = []
    causes = []
    causesDesire = []
    hasPrerequisite = []
    for e in k:
      temp = e.split()[1:]
      to_add = " ".join(temp)
      if e.startswith("RelatedTo"):
        relatedTo.append(to_add)
      elif e.startswith('Causes'):
        causes.append(to_add)
      elif e.startswith('CausesDesire'):
        causesDesire.append(to_add)
      elif e.startswith("HasPrerequisite"):
        hasPrerequisite.append(to_add)
    know = ''
    event = event[0].capitalize() + event[1:]
    if len(relatedTo) >0:
      know += event + " relates to " + ", ".join(relatedTo) + ". "
    if len(causes) >0:
      know += event + " causes " + ", ".join(causes) + ". "
    if len(causesDesire) >0:
      know += event + " causes someone want " + ", ".join(causesDesire) + ". "
    if len(hasPrerequisite) >0:
      know += event + " needs the following to happen beforehand: " + ", ".join(hasPrerequisite) + ". "
    knowledge.append(know)
  return knowledge