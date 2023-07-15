import numpy as np
import pandas as pd
import re
import jsonlines
from tqdm import tqdm

import openai
from constant import openai_key

from datasets import load_dataset
from transformers import AutoTokenizer

openai.api_key = openai_key
model_engine = "gpt-4" 
n_sample = 200
words2sent = AutoTokenizer.from_pretrained("bert-large-cased").convert_tokens_to_string

dataset_name = "sst2"
label_texts = ["sentiment: negative", "sentiment: positive"]
dataset_train = load_dataset(f"glue", dataset_name)["train"]
dataset_train = np.random.choice(dataset_train, n_sample, replace=False)

def create_query(sentence, label_text, other_label_text):
     return f'''"{sentence}"
Please think step by step:
1. What are some other attributes of the above sentence except \"{label_text}\"?
2. How to write a similar sentence with these attributes and \"{other_label_text}\"?
3. Write such a sentence without any other explanation.'''

def decode_response(response):
    for line in response.split("\n"):
        if line.startswith("3."):
            return line[2:].strip().strip("\"")
        
def attribute_manipulate(data):
    
    items = dict()

    sentence = data['sentence']
    sentence = words2sent(sentence.split())
    label_text = label_texts[data['label']]
    other_label_texts = label_texts.copy()
    other_label_texts.remove(label_text)

    items[label_text] = sentence

    for other_label_text in other_label_texts:

        query = create_query(sentence, label_text, other_label_text)

        try:
            response = openai.ChatCompletion.create(
                        model=model_engine,
                        temperature=0.,
                        messages=[
                            {"role": "user", "content": query},
                        ],
                        ).choices[0]['message']["content"]
        except:
            response = ""

        other_sentence = decode_response(response)

        if other_sentence is not None:
            items[other_label_text] = other_sentence
            items[f"{other_label_text} (response)"] = response
            
    return items

np.random.shuffle(dataset_train)

with jsonlines.open(f"datasets/{dataset_name}.cotam.2.train.jsonl", "w") as writer:

    for idx, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
        
        items = attribute_manipulate(data)
        
        writer.write(items)