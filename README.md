# jp_news_cls
This project is based on RoBERT to fine-tune and classify Japanese news using two different tokenizers.

## Overview
This is a simple comparative test using RoBERTa, and it's my first time using Hugging Face's Transformers library. The project serves as an easy introduction to building a model with a pre-trained model through Hugging Face.

This program also includes some code snippets inspired by a Chinese tutorial found here. I would like to express my thanks to the original author.

## Installation
Before you start, make sure you have the necessary libraries installed. You can do this via pip or conda：
`pip install torch transformers pandas scikit-learn`
or
`conda install torch transformers pandas scikit-learn`
in your anaconda env.

## Dataset
You can either:

Download an open-source dataset and write your own function to process the data.
Or, use the Japanese news dataset available under the Creative Commons license [here](https://creativecommons.org/licenses/by-nd/2.1/jp/)

(please read the license)

## How to use 
Convert that japanese dataset into the required format:
`python txt_to_csv.py`

Transform the labels:
`python trans_label.py`

Train and test the model:
`python train_robert.py`

or you could make a function to read the file to create the dataset.

## Conclusion
This project can serve as a foundation for fine-tuning models for other tasks as well. Feel free to experiment and extend its capabilities.
