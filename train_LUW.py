import unicodedata
import re
import pandas as pd
import torch
import os

from tokenizers import Tokenizer,models,pre_tokenizers,normalizers,decoders,trainers
from transformers import RemBertTokenizerFast,AutoTokenizer,get_scheduler,AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils import shuffle

# make the model and tokenizer
# AutoModelForSequenceClassification is a function specially designed for text classification.
# the different with the previous code is that the LUW be used as tokenizer 
# as you can see, the loss is a little bigger than the previous code
# but the test result is not bad, I will test them with other dataset in the future
# you also could change the num_labels to the number of your label

tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-japanese-luw-upos")
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


# step:  10   loss: 0.13954393118619918
# step:  20   loss: 0.13964745327830314
# step:  30   loss: 0.13929008295138676
# step:  40   loss: 0.13970570228993892
# step:  50   loss: 0.13944958180189132
# step:  60   loss: 0.13922391360004743
# step:  70   loss: 0.1392746395298413
# step:  80   loss: 0.139160917699337
# step:  90   loss: 0.13894355835186112
# step:  100   loss: 0.1388008949905634
# step:  110   loss: 0.13865940672430124
# step:  120   loss: 0.1383491560195883
# step:  130   loss: 0.13842499594275767
# step:  140   loss: 0.1384864976895707
# step:  150   loss: 0.13862608686089517
# step:  160   loss: 0.13856650316156446
# step:  170   loss: 0.13859204601715594
# step:  180   loss: 0.1386505912989378
# step:  190   loss: 0.13860745167261676
# step:  200   loss: 0.13859345983713867
# step:  210   loss: 0.1385004225586142
# step:  220   loss: 0.1384103136645122
# step:  230   loss: 0.1385161407939766
# step:  240   loss: 0.13853983404114842
# step:  250   loss: 0.13850072667002677
# step:  260   loss: 0.13842260668484063
# step:  270   loss: 0.13845941708595663
# step:  280   loss: 0.13849667898778403
# step:  290   loss: 0.13846129448763256
# step:  300   loss: 0.1384996289263169
# step:  310   loss: 0.13847313828526003
# step:  320   loss: 0.13837016054894774
# step:  330   loss: 0.13843489399913586
# step:  340   loss: 0.1384575102916535
# step:  350   loss: 0.13841540926269122
# step:  360   loss: 0.13839858217785755
# step:  370   loss: 0.13837410185385396
# step:  380   loss: 0.1383862859520473
# step:  390   loss: 0.13832726900776227
# step:  400   loss: 0.13829317854717374
# step:  410   loss: 0.13825558000584928
# step:  420   loss: 0.1382718673951569
# step:  430   loss: 0.1382480853345505
# step:  440   loss: 0.13819563255052675
# step:  450   loss: 0.13815274452169737
# step:  460   loss: 0.1381516316824633
# avg_loss: 2.210336681301837