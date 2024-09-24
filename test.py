import os
import re
import torch

from sklearn.utils import shuffle
import pandas as pd

from transformers import AutoModelForSequenceClassification,AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_scheduler
from torch.optim import AdamW

# please write your test word here
# like this: test_word = "いい天気から、散歩しましょう。"
test_word = ""

# choose model, and comment out another
model = "./LUW_model_save"
model = "./robert_model_save"

# make the model you trained
# AutoModelForSequenceClassification is a function specially designed for text classification.
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=9)
test = tokenizer(test_word,return_tensors="pt",padding="max_length",max_length=100)
model.eval()

# predict the label
with torch.no_grad():  
    outputs = model(test["input_ids"], 
                    token_type_ids=None, 
                    attention_mask=test["attention_mask"])
    
# argmax to get the label
pred_flat = np.argmax(outputs["logits"],axis=1).numpy().squeeze()
print(pred_flat.tolist())
