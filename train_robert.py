import os
import re
import torch

from sklearn.utils import shuffle
import pandas as pd

from transformers import AutoModelForSequenceClassification,AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_scheduler
from torch.optim import AdamW

# make the model and tokenizer
# AutoModelForSequenceClassification is a function specially designed for text classification.
# the rebert in AutoModelForSequenceClassification 
# is a pretrained model for text classification I downloaded from huggingface for japanese text
# you also could change the num_labels to the number of your label

tokenizer = AutoTokenizer.from_pretrained("robert", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("robert", num_labels=9)

# this label dict is based on the 「livedoor ニュース」's opensource dataset
# datset website at here https://creativecommons.org/licenses/by-nd/2.1/jp/

label_dic = {
 'dokujo-tsushin':0,
 'it-life-hack':1,
 'kaden-channel':2,
 'livedoor-homme':3,
 'movie-enter':4,
 'peachy':5,
 'topic-news':6,
 'smax':7,
 'sports-watch':8}

# you also could use other way to get the data not just csv file
def get_csv_train_data(file, label_dic):
    data = pd.read_csv(file)
    content = data["content"].to_list()
    label = data["label"].to_list()
    return content, label


content, label = get_csv_train_data("processed_data.csv", label_dic)
data = pd.DataFrame({"label":label, "content":content})

# shuffle the data for better training
data = shuffle(data)

# I trained the model with my cpu instead of gpu, so I just use 8000 data for training for saving time
# you also could change the 
train_data = tokenizer(data.content.to_list(), padding = "max_length", max_length = 100, truncation=True ,return_tensors = "pt")
train_label = data.label.to_list()

# you could change the batch size to a bigger number if you have a better gpu
batch_size = 16

# make the dataloader
# TensorDataset is a dataset wrapping tensors, and DataLoader is a iterator for the dataset
train = TensorDataset(train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
train_sampler = RandomSampler(train)
train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size)

# define the optimizer
# the AdamW is a great enought, but you could also use other optimizer like SGD
optimizer = AdamW(model.parameters(), lr=1e-4)

# define the learning rate(lr) and the number of epochs(num_epochs)
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# move the model to gpu if you have one
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# start training
# the grad will be zeroed after each batch for pytorch will accumulate the grad
# b_input_ids is the input data, 
# token_type_ids is choose the type of the token,
# b_labels is the label for the input data
# attention_mask is the mask for the input data to let the model pay attention to the right place
# the labels is to calculate the loss
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            print("step: ",step, "  loss:",total_loss/(step*batch_size))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()        
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        loss = outputs.loss       
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)      
    print("avg_loss:",avg_train_loss)

for param in model.parameters():
    param.data = param.data.contiguous()

# save the model to the place you want
output_dir = "./model_save/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"the model has been saved at {output_dir}")

# output
# step:  10   loss: 0.14297897815704347
# step:  20   loss: 0.14021991714835166
# step:  30   loss: 0.13583668768405915
# step:  40   loss: 0.12884119246155024
# step:  50   loss: 0.12322387173771858
# step:  60   loss: 0.11909563864270846
# step:  70   loss: 0.1128561477043799
# step:  80   loss: 0.10907359388656915
# step:  90   loss: 0.10709343035187986
# step:  100   loss: 0.105511543340981
# step:  110   loss: 0.10288670628585599
# step:  120   loss: 0.10097328322008252
# step:  130   loss: 0.09892801344394683
# step:  140   loss: 0.09749811395470585
# step:  150   loss: 0.09608819703261058
# step:  160   loss: 0.09384309817105532
# step:  170   loss: 0.09198731663910782
# step:  180   loss: 0.09048353150073025
# step:  190   loss: 0.0891540805564115
# step:  200   loss: 0.08798897605389357
# step:  210   loss: 0.0871463235822462
# step:  220   loss: 0.08621979679235003
# step:  230   loss: 0.0851140000897905
# step:  240   loss: 0.08409939355527361
# step:  250   loss: 0.0833121796399355
# step:  260   loss: 0.08214435018599034
# step:  270   loss: 0.0812452700678949
# step:  280   loss: 0.08009868268189686
# step:  290   loss: 0.07930000762872655
# step:  300   loss: 0.07847068071986238
# step:  310   loss: 0.07747206924062583
# step:  320   loss: 0.07678444603807293
# step:  330   loss: 0.07644861573635629
# step:  340   loss: 0.07577480015290135
# step:  350   loss: 0.07509234929723399
# step:  360   loss: 0.07428160163884362
# step:  370   loss: 0.07372902982138299
# step:  380   loss: 0.07330477222015983
# step:  390   loss: 0.07288432971407206
# step:  400   loss: 0.07263061465229839
# step:  410   loss: 0.07198806786682548
# step:  420   loss: 0.07151516026684217
# step:  430   loss: 0.07085220655209797
# step:  440   loss: 0.07016334741651505
# step:  450   loss: 0.06967375812534657
# step:  460   loss: 0.06916093225894576