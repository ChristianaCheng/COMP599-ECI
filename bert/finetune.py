from datasets import load_dataset,DatasetDict
import evaluate
import numpy as np
import torch
from transformers import (AutoTokenizer,
AutoModelForSequenceClassification,
TrainingArguments, 
Trainer,
DataCollatorWithPadding,
EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

data_files = {
    "train":"../data/EventStoryLine_train_v0.9.json",
    "test":"../data/EventStoryLine_test_v0.9.json"
}
dataset = load_dataset("json", data_files=data_files,cache_dir='~/scratch/comp599/bert_all/')
add_knowledge = True
def preprocess_function(batch):
    sent1_id,sent2_id = batch['sentence1_id'],batch['sentence2_id']
    if sent1_id == sent2_id:
      sentence = batch["marked_sent1"]
      #print(sentence)
      model_inputs = tokenizer(
              sentence,
              max_length=128,
              padding='max_length',
              truncation=True,
          )

    else:
      sentence1,sentence2 = batch["marked_sent1"],batch["marked_sent2"]
      #print(sentence1,sentence2)
      model_inputs = tokenizer(
              sentence1,
              sentence2,
              max_length=128,
              padding='max_length',
              truncation=True,
          )
    return model_inputs

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  preds = np.argmax(logits, axis=-1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
  return {'f1': f1,'precision': precision,'recall': recall}

# create 5-fold
folds = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
splits = folds.split(np.zeros(dataset["train"].num_rows), dataset["train"]["label"])
fold_dataset = None
column_names = ['event1_id', 'event2_id', 'event1', 'event2', 'sentence1_id', 'sentence2_id', 'sentence1', 'sentence2', 'relation', 'topic', 'ducument', 'article', 'marked_sent1', 'marked_sent2','intra']
split = 0
r = []
# do the training with 5-fold cross validation
for train_idx, val_idx in splits:
    split += 1
    print(f"***********NOW: FOLD {split}***********")
    fold_dataset = DatasetDict({
    "train":dataset["train"].select(train_idx),
    "validation":dataset["train"].select(val_idx),
    "test":dataset["test"]
})
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # add special event marker to the vocab of the model
    special_tokens = ["<e1>","</e1>","<e2>","</e2>"]
    if add_knowledge:
      special_tokens.extend(['<RelatedTo>','<Causes>','<CausesDesire>','<HasPrerequisite>','<s>'])
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    train_raw = fold_dataset['train']
    val_raw = fold_dataset['validation']
    
    train_dataset = train_raw.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=column_names
                    )
    val_dataset = val_raw.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=column_names,
                    )
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # training
    training_args = TrainingArguments(
    output_dir="~/scratch/comp599/bert_all/"+str(split),
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    lr_scheduler_type= "linear",
    evaluation_strategy = 'steps',
    eval_steps= 500,
    save_strategy = 'steps',
    save_steps= 500,
    logging_steps = 500,
    warmup_steps=500, 
    seed = 42,
    load_best_model_at_end= True,
    metric_for_best_model= "f1",
    greater_is_better= True,
    save_total_limit= 3,

)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        
    )

    trainer.train()
    # Evaluation
    test_raw = fold_dataset['test']
    test_dataset = test_raw.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=column_names,
                    )
    results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Best result from fold {split} {results})")
    r.append(results)
print(r)
