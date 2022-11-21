# Datasets
<h2> EventStoryLine </h2>
<h4> Steps to reproduce </h4>
  <ul>
    <li> <a href = "https://github.com/tommasoc80/EventStoryLine.git"> Clone the official Event Story Line github repo </a></li>
    <li>Run create_raw.py to get raw data in pickle format with specified version</li>
    <li>Run create_corpus.py to get the causal dataset in json format. Inter-sentential and intra-sentential data are stored in two different files.
          Fields in the .json file: <br>
          1. event1_id/event2_id: the unique id of the event in the corresponding sentence (begins at 0)<br>
          2. event1/event2: the event token<br>
          3. sentence1_id/sentence2_id: the unique id of the sentence in the article (begins at 0)<br>
          4. sentence1/sentence2: the sentence that involves the event<br>
          5. relation: the target; could be 'FALLING_ACTION', 'null' or 'PRECONDITION' in our processed corpus<br>
          6. document: the article that the events/sentences belong to<br>
          7. topic: the topic that the article belongs to. In total, there are 22 topics.<br>
          8. article: the article contains all information of the document, containing tuples of (word_id, sentence_id, word_id_in_the_sentence, word) for the whole article.<br></li>
     <li> TODO: post-process: remove duplicates <\li>
    
  </ul>
<h2> Causal-Timebank </h2>
