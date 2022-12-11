import openai
import json
import time
# Load your API key from an environment variable or secret management service
openai.api_key = open("key.txt","r")

intra = "../t0_data/EventStoryLine_test_intra_v0.9_conceptnet.json"

option = "step_by_step"
mode = "w"
preds = open(f"../openai_results/openai_{option}.json",mode)
counter = 0

with open(intra,"r",encoding = 'utf-8') as f:
    for l in f:
        time.sleep(3)
        counter += 1
        line = json.loads(l)
        e1 = line['event1']
        e2 = line['event2']
        sentence = line['sentence1']
        if option == 'answer_guide':
          guide = "Answer Yes or No."
          prefix = f"Is there a causal relationship between the word \"{e1}\" and \"{e2}\" in the following sentence? {guide}\n{sentence}\n"
          max_tokens = 3
        elif option == 'answer_guide_flipped':
          guide = "Answer Yes or No."
          prefix = f"Is there a causal relationship between the word \"{e1}\" and \"{e2}\" in the following sentence?\n{sentence}\n{guide}"
          max_tokens = 3

        elif option == 'step_by_step':
          steps = f"Please think about the following questions and answer with Yes or No in the form of 1.X 2.X 3.X\n1. Does \"{e1}\" cause \"{e2}\" in this sentence? \n2. Does \"{e2}\" cause \"{e1}\" in this sentence?\n3. Is a causal relationship between \"{e1}\" and \"{e2}\" in the sentence?\n1."
          answer_hint= "If the answers for the two first questions are No, then the third answer should be No.If one of the answers for the first two questions is Yes, then the third answer should be Yes too."
          prefix = f"{sentence}\n{steps}"
          max_tokens = 15

        elif option == 'hint':
          guide = "Answer Yes or No."
          question = f"Is there a causal relationship between the word \"{e1}\" and \"{e2}\" in the following sentence?"
          hint = "Note that a causal relationship could be explicit which is introduced by signals such as because, by, from, for, among others, as, etc., or implicit.But the presence of causal signals does not necessarily indicate a causal relationship between the two events."
          prefix = f"{question}{sentence}{hint}{guide}"
          max_tokens = 3
        elif option == 'hint_flipped':
          guide = "Answer Yes or No."
          question = f"Is there a causal relationship between the word \"{e1}\" and \"{e2}\" in the following sentence?"
          hint = "We know that a causal relationship could be explicit which is introduced by signals such as because, by, from, for, among others, as, etc., or implicit.But the presence of causal signals does not necessarily indicate a causal relationship between the two events."
          prefix = f"{hint}\n{question}{guide}{sentence}"
          max_tokens = 3

        elif option == 'external_knowledge':
          external_knowledge = line['external_knowledge']
          question = f"Is there a causal relationship between the word \"{e1}\" and \"{e2}\" in the following sentence?"
          guide = "Answer Yes or No."
          prefix = f"{question}{sentence}We know that:{external_knowledge}\n{guide}"
          max_tokens = 3


        response = openai.Completion.create(model="text-davinci-003", 
                                          prompt=prefix, 
                                          temperature=0, 
                                          max_tokens=max_tokens)
        pred = response.choices[0]
        print(("1."+response.choices[0].text).strip())
        print("-"*50)

        json.dump(pred,preds)
        preds.write('\n')
print(counter)
