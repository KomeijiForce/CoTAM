import numpy as np
import pandas as pd
import jsonlines

from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

import torch
from transformers import AutoTokenizer, AutoModel

words2sent = AutoTokenizer.from_pretrained("bert-large-cased").convert_tokens_to_string

K = 10
batch_size = 128
device_idx = "cuda:0"
fname = "datasets/sst2.cotam.train.jsonl"
encoder_name = "princeton-nlp/sup-simcse-roberta-large"
dataset_name = "sst2"
label_texts = ["sentiment: negative", "sentiment: positive"]

device = torch.device(device_idx)
tok = AutoTokenizer.from_pretrained(encoder_name)
encoder = AutoModel.from_pretrained(encoder_name).to(device)
dataset_test = load_dataset("glue", "sst2")["validation"]

def load_cotam(fname, label_texts, K):

    dataset_train = []

    dataset = [items for items in jsonlines.open(fname) if len(items) == 2 * len(label_texts) - 1]
    
    label2dataset = defaultdict(list)
    
    for data in dataset:
        label2dataset[list(data.keys())[0]].append(data)

    dataset = [items for label_text in label_texts for items in np.random.choice(label2dataset[label_text], K, replace=False)]
    
    np.random.shuffle(dataset)

    for items in dataset:
        for label_text in label_texts:
            text = items[label_text]
            dataset_train.append({"text":text, "label":label_text})
            
    return dataset_train

accs = []

for run in range(10):

    dataset_train = load_cotam(fname, label_texts, K)

    clusters = {label_text:[] for label_text in label_texts}

    for data in dataset_train:
        clusters[data["label"]].append(data["text"])

    with torch.no_grad():

        centers = dict()

        for label_text in label_texts:

            texts = clusters[label_text]

            centers[label_text] = encoder(**tok(texts, padding=True, return_tensors="pt", truncation=True).to(device)).pooler_output.mean(0)

    res = []

    with torch.no_grad():

        for idx in range(0, len(dataset_test), batch_size):
            labels = [text for text in dataset_test[idx:idx+batch_size]["label"]]
            texts = [text for text in dataset_test[idx:idx+batch_size]["sentence"]]
            reprs = encoder(**tok(texts, padding=True, return_tensors="pt", truncation=True).to(device)).pooler_output

            distances = torch.stack([((reprs - centers[label_text].unsqueeze(0))**2).mean(-1) for label_text in label_texts], -1)

            predictions = distances.argmin(-1)

            res.extend(np.array(labels) == predictions.detach().cpu().numpy())

    acc = np.mean(res)
    accs.append(acc)

    print(f"#Run: {run+1} #Accuracy: {np.mean(accs)*100:.4}%")