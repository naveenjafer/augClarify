import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
import numpy as np
import time
import datetime

def tokenize_and_format(sentences):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []

  # For every sentence...
  for sentence in sentences:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sentence,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          padding = 'max_length',
                          truncation = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])
  return input_ids, attention_masks

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_text_label_list(dataList, labelToIndexMap=None):
    textList = []
    labelList = []
    if labelToIndexMap == None:
        labelToIndexMap = {}
    counter = 0
    for dataObj in dataList:
        textList.append(dataObj.text)
        
        if dataObj.label not in labelToIndexMap:
            labelToIndexMap[dataObj.label] = counter
            counter += 1
        labelList.append(labelToIndexMap[dataObj.label])
    
    return textList, labelList, labelToIndexMap

def get_text_label_actual_list(dataList):
    textList = []
    labelList = []

    counter = 0
    for dataObj in dataList:
        textList.append(dataObj.text)
        labelList.append(dataObj.label)
    
    return textList, labelList