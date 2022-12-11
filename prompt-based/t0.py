from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "bigscience/T0_3B"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

task_description = "This is an event causality classifier to identify if there is a causal relationship between two events in the sentences.\n"
#pos_ex = "Is there a causal relationship between the word 'arrested' and 'killed' in the following sentence?\nThe police arrested him because he killed someone.\nYes\n"
pos_ex = "Is there a causal relationship between the word 'clashes' and 'enraged' in the following sentence?\nProtesters enraged over the fatal shooting of a teenager by police poured into Brooklyn streets for a third straight night Wednesday , pitching bricks , bottles and garbage in furious clashes with cops .\nYes\n"
neg_ex = "Is there a causal relationship between the word 'causing' and 'arrested' in the following sentence?\nJenkin, who was arrested at Millom Pier at around 9.35am, is also facing a charge of causing unnecessary suffering to an animal in relation to the dog's death.\nNo\n"

preds = open("test0.9_preds.txt",'w')
intra = "data/EventStoryLine_test_intra_v0.9.json"
with open(intra,"r",encoding = 'utf-8') as f:
    for line in f:
        e1 = line['event1']
        e2 = line['event2']
        sentence = line['sentence1']
        inp = f"Is there a causal relationship between the word '{e1}' and '{e2}' in the following sentence?\n{sentence}\n"
        #for i in [task_description+pos_ex+neg_ex+inp, task_description+inp, inp]:
        # print(i)
        i = task_description+pos_ex+neg_ex+inp
        inputs = tokenizer.encode(i, return_tensors="pt")
        inputs = inputs.to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(inputs)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(pred)
        preds.write(pred)

    print("FINISHED")




# event1 = "hearing"
# event2 = "held"
# sentence = "A plea and case management hearing will be held on September 6."
