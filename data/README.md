# Datasets
<h2> EventStoryLine </h2>
<h4> Steps to reproduce </h4>
  <ul>
    <li> <a href = "https://github.com/tommasoc80/EventStoryLine.git"> Clone the official Event Story Line github repo </a></li>
    <li>Run create_raw.py to get raw data in pickle format with specified version</li>
    <li>Run create_corpus.py to get the causal dataset in json format. Inter-sentential and intra-sentential data are stored in two different files.
            Fields in the .json file: <br>
            1. event1_id/event2_id: the unique id of the event in the corresponding sentence (begins at 0)<br>
            2. event1/event2: the event token; NOTE: 1. one event could contain multiple continous tokens) 2.Causality has order(e.g. A causes B and B causes A is different); event1 is the "source" and event2 is the "target" but event1 does not necessary appear before event2.; the order is determined by event ids <br>
            3. sentence1_id/sentence2_id: the unique id of the sentence in the article (begins at 0)<br>
            4. sentence1/sentence2: the sentence that involves the event, event1 is in sentence1 and event2 is in sentence2. NOTE that sentence1 might appear later in the article than sentence2. The order is determined by the sentence ids.<br>
            5. relation: the target; Could be 'FALLING_ACTION', 'null' or 'PRECONDITION' in our processed corpus; 'null' indicates no causality and both 'FALLING_ACTION' and 'PRECONDITION' indicates the causal relations between the two events. <br>
            6. document: the file path of the article that the events/sentences belong to<br>
            7. topic: the topic that the article belongs to. In total, there are 22 topics.<br>
            8. article: the article contains all information of the document, containing tuples of (word_id, sentence_id, word_id_in_the_sentence, word) for the whole article.<br></li>
     <br>
    <i>NOTE: The json files are too big to upload on github. Only an example was given (in zip format). Follow the instructions above locally can get the processed causal inter- and intra- sentential causal corpus.</i>
    
  </ul>
