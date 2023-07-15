import numpy as np
import pandas as pd
import re
import jsonlines
from tqdm import tqdm

from collections import defaultdict

import torch
from torch import nn
from torch.optim import Adam

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

from transformers import logging

logging.set_verbosity_error()

words2sent = AutoTokenizer.from_pretrained("bert-large-cased").convert_tokens_to_string

device_idx = 'cuda:0'
fname = "datasets/sst2.cotam.train.jsonl"
label_texts = ["sentiment: negative", "sentiment: positive"]
K = 10

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

def load_data(fname, split, K):
    
    label2dataset = defaultdict(list)

    for data in load_dataset("glue", fname)[split]:
        text, label = data["sentence"], data["label"]
        label = label_texts[label]
        text = words2sent(text.split())
        label2dataset[label].append({"text":text, "label":label})
        
    dataset = []
        
    for label in label2dataset:
        
        dataset.extend(np.random.choice(label2dataset[label], K, replace=False) if K is not None else label2dataset[label])
        
    np.random.shuffle(dataset)
        
    return dataset

class RobertaClassifier:

    def __init__(self, model_name='bert-large-uncased', device='cuda:4', num_labels=2, learning_rate=1e-5, eps=1e-6, betas=(0.9, 0.999)):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = Adam([p for p in self.classifier.parameters()], lr=learning_rate, eps=eps, betas=betas)

    def train(self, dataset, batch_size=16):

        for idx in range(0, len(dataset), batch_size):
            tups = dataset[idx:idx + batch_size]
            texts = [tup["text"] for tup in tups]
            golds = [label_texts.index(tup["label"]) for tup in tups]

            inputs = self.tok(texts, padding=True, return_tensors='pt').to(self.device)
            scores = self.classifier(**inputs)[-1]
            golds = torch.LongTensor(golds).to(self.device)

            self.classifier.zero_grad()

            loss = self.criterion(scores, golds).mean()

            loss.backward()

            self.optimizer.step()

    def evaluate(self, dataset, batch_size=16):
        
        scoreboard = torch.BoolTensor([]).to(self.device)
        losses = torch.FloatTensor([]).to(self.device)
        
        with torch.no_grad():
            for idx in range(0, len(dataset), batch_size):
                tups = dataset[idx:idx + batch_size]
                texts = [tup["text"] for tup in tups]
                golds = [label_texts.index(tup["label"]) for tup in tups]

                inputs = self.tok(texts, padding=True, return_tensors='pt').to(self.device)
                scores = self.classifier(**inputs)[-1]
                preds = scores.argmax(-1)
                golds = torch.LongTensor(golds).to(self.device)
                
                losses = torch.cat([losses, self.criterion(scores, golds)], 0)
                scoreboard = torch.cat([scoreboard, (preds == golds)], 0)
                acc = scoreboard.float().mean().item()
                
                pred_labels = [label_texts[pred.item()] for pred in preds]
                
        return acc
    
dataset_test = load_data("sst2", "validation", None)

accs = []

for run in range(10):
    
    dataset_train = load_cotam(fname, label_texts, K)

    classifier = RobertaClassifier(model_name='roberta-large', num_labels=len(label_texts), learning_rate=1e-5, device=device_idx)

    for idx in range(32):
        classifier.train(dataset_train, 16)

    acc = classifier.evaluate(dataset_test, 16)

    accs.append(acc)

    print(f"#Run: {run+1} #Accuracy: {np.mean(accs)*100:.4}%")