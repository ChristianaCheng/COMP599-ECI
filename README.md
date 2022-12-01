ECI
- BiLSTM
- Bert
- T0

## knowledge based.py
- Knowledge Retrieving from Conceptnet
- knowledge BERT encoding
- Masking Reasoner: by adding a special token [MASK] to replace the event in sent1 and sent2. Default is True.
1. Load dataset from json file
2. The data structure: dict[list[dict]]]. For instance, {'intra': [{datapoint1},{datapoint2}...], 'inter': [{datapoint1},{datapoint2}...]}
