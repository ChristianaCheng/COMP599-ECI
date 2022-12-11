import json
import requests
import spacy
from conceptnet_utils import wordify_knowledge,crawl_concept_net,linear_knowledge

nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")

fname = f"t0_data/EventStoryLine_test_intra_v0.9.json"
outfile = f"t0_data/EventStoryLine_test_intra_v0.9_conceptnet.json"
outpath = open(outfile,"w",encoding='utf-8')

with open(fname,"r",encoding='utf-8') as f:
	for line in f:
		doc = json.loads(line)
		external_knowledge = (" ".join(wordify_knowledge(doc))).strip()
		doc.update(
			{"external_knowledge":external_knowledge}
			)
		json.dump(doc,outpath,ensure_ascii=False)
		outpath.write('\n')

