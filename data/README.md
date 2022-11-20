# Datasets
## EventStoryLine

The original corpus is structured in folders; each folder corresponds to a topic from the ECB corpus. The file is in XML-labeled annotation format. Therefore, we need to use lxml to read the corpus.
Annotation Information from the corpus:
<token>:
t_id: token unique id: starts at 1
sentence: sentence id: start at 0
number: t_id per sentence: start at 0
2)<Action_*> & <NEG_Action>:
The tag to contain the label of event type, where the father node is <Markables>
m_id: unique markable id for the event
<TIME_*>: temporal expression
DCT: TF to identify the doc creation time temporal expression
value: the normalized value for temporal expression
anchorTimeID: markable id of the anchor Timex used to normalize the value/resolve the current temporal expressions
<token anchor>: Identify the corresponding tokens with the father node: <ACTION_>, <NEG_ACTION_>, <TIME_*>
5)<TLINK>: link used to annotate the temporal Relation (Father node: < Relation>)
r_id: unique id of the link tag
contextualModality : factuality value of the TLINK - not annotated at the moment.
relType: temporal relation value of the TLINK tag
<PLOT_LINK>: Annotated explanatory relation
r_id: unique id of the link tag
relType: Precondition vs. Falling action
SIGNAL: markable ID introduces an explicit causal relation
CAUSE: True vs False
CAUSED_BY: True vs. False to identify if the event source IS_CAUSED the event target
The data is stored in a JSON file.


2. Causal-Timebank
