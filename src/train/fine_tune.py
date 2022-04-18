import torch
from helpers import tokenize_and_format, flat_accuracy, get_text_label_list
from data_load import load_intent_examples
# Confirm that the GPU is detected
import pandas as pd
from consts import banking77_conf, art_conf
import os
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import sys
import random
from datetime import datetime
def loadData(folderName, folderIdentifier):
    conf = {}
    if folderIdentifier == "banking77_conf":
        conf = banking77_conf
    
    elif folderIdentifier == "art_conf":
        conf = art_conf

    train_data = load_intent_examples(os.path.join(folderName, conf["train"]))
    dev_data = load_intent_examples(os.path.join(folderName, conf["dev"]))
    test_data = load_intent_examples(os.path.join(folderName, conf["test"]))

    return train_data, dev_data, test_data
    #indexToLabelMap = {v: k for k, v in labelToIndexMap.items()}

folderName = sys.argv[1]
folderIdentifier = sys.argv[2]

inputFolderModified = folderName.replace("/","_")

if not os.path.isdir("models"):
    os.mkdir("models")

if not os.path.isdir(os.path.join("models",inputFolderModified)):
    os.mkdir(os.path.join("models",inputFolderModified))

now = datetime.now()
modelFolder = os.path.join("models", inputFolderModified, now.strftime("%m_%d_%Y_%H_%M_%S"))
os.mkdir(modelFolder)

# Get the GPU device name.
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    device = torch.device("cuda")
else:
    device = "cpu"



train_data, dev_data, test_data = loadData(folderName, folderIdentifier)
train_text, train_label, label_to_index_map = get_text_label_list(train_data)
train_input_ids, train_attention_masks = tokenize_and_format(train_text)
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_label)

dev_text, dev_label, dev_map = get_text_label_list(dev_data, label_to_index_map)
dev_input_ids, dev_attention_masks = tokenize_and_format(dev_text)
dev_input_ids = torch.cat(dev_input_ids, dim=0)
dev_attention_masks = torch.cat(dev_attention_masks, dim=0)
dev_labels = torch.tensor(dev_label)

test_text, test_label, test_map  = get_text_label_list(test_data, label_to_index_map)
test_input_ids, test_attention_masks = tokenize_and_format(test_text)
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_label)
'''
print("Training data size: ", len(train_text))
print(train_text[0:4])

print("Number of labels: ", len(label_to_index_map.keys()))
print(train_label)
print(dev_label)
print("Labels: ", label_to_index_map)
print("******************")
print("Dev label map", dev_map)
print("*****************")
print("Testlabel map", test_map)
'''
train_set = [(train_input_ids[i], train_attention_masks[i], train_labels[i]) for i in range(len(train_input_ids))]
random.shuffle(train_set)
dev_set = [(dev_input_ids[i], dev_attention_masks[i], dev_labels[i]) for i in range(len(dev_input_ids))]
test_set = [(test_input_ids[i], test_attention_masks[i], test_labels[i]) for i in range(len(test_input_ids))]

model = BertForSequenceClassification.from_pretrained(
"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
num_labels = len(label_to_index_map.keys()), # The number of output labels.   
output_attentions = False, # Whether the model returns attentions weights.
output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

batch_size = 4
optimizer = AdamW(model.parameters(),
                lr = 2e-5, # args.learning_rate - default is 5e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
epochs = 50

def get_validation_performance(val_set):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0

    num_batches = int(len(val_set)/batch_size) + 1

    total_correct = 0

    for i in range(num_batches):

      end_index = min(batch_size * (i+1), len(val_set))

      batch = val_set[i*batch_size:end_index]
      
      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])
      
      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)
        
      # Tell pytorch not to bother with constructing the compute graph during
      # the forward pass, since this is only needed for backprop (training).
      with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the number of correctly labeled examples in batch
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        #print(labels_flat)
        #print(pred_flat)
        #quit(1)
        num_correct = np.sum(pred_flat == labels_flat)
        total_correct += num_correct
        
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_correct / len(val_set)
    return avg_val_accuracy


for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    num_batches = int(len(train_set)/batch_size) + 1

    for i in range(num_batches):
      end_index = min(batch_size * (i+1), len(train_set))

      batch = train_set[i*batch_size:end_index]

      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])

      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)

      # Clear the previously calculated gradient
      model.zero_grad()        

      # Perform a forward pass (evaluate the model on this training batch).
      outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
      loss = outputs.loss
      logits = outputs.logits

      total_train_loss += loss.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Update parameters and take a step using the computed gradient.
      optimizer.step()
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set. Implement this function in the cell above.
    print(f"Total loss: {total_train_loss}")
    val_acc = get_validation_performance(dev_set)
    train_acc = get_validation_performance(train_set)
    print(f"Training accuracy: {train_acc}")
    print(f"Validation accuracy: {val_acc}")
    
print("")
print("Training complete!")

print("Running on test set")
get_validation_performance(test_set)
model.save_pretrained(modelFolder)
# save model


#Next, you can use the model.save_pretrained("path/to/awesome-name-you-picked") method. This will save the model, with its weights and configuration, to the directory you specify. Next, you can load it back using model = .from_pretrained("path/to/awesome-name-you-picked").
