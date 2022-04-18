# takes as input a folder from data and generates the training data for it.
import os
import sys
from utils import seed_everything
from samplingMethods import random_sampling
from src.train.data_load import load_intent_examples
from src.train.consts import banking77_conf, art_conf
from src.train.helpers import get_text_label_actual_list
import csv

conf = {}
def cleanup(train_text):
    for index, text in enumerate(train_text):
        train_text[index] = text.replace(",", "")
        train_text[index] = train_text[index].replace("_", "")
    return train_text

def buildInput(train_text_sampled, train_label):
    inputList = []
    for item in zip(train_text_sampled, train_label):
        textScrambled = " | ".join(item[0])
        inputList.append(item[1] + " _ " + textScrambled)
    return inputList

def writeFile(sourceTextList, targetTextList, outputFolderName):
    if not os.path.isdir(os.path.join("dataAugT5", outputFolderName)):
        os.mkdir(os.path.join("dataAugT5", outputFolderName))

    with open(os.path.join("dataAugT5", outputFolderName, "training.csv"), 'w') as f:
        writer = csv.writer(f)
        #header write
        writer.writerow(['target', 'source'])
        for item in zip(sourceTextList, targetTextList):
            writer.writerow([item[1], item[0]])
    
def process(inputFolder, outputFolderName):
    train_data = load_intent_examples(os.path.join(inputFolder, conf["train"]))
    train_text, train_label = get_text_label_actual_list(train_data)
    train_text = cleanup(train_text)

    train_text_sampled = [random_sampling(item,0.7) for item in train_text]

    sourceTextList = buildInput(train_text_sampled, train_label)
    targetTextList = train_text # might have to change this

    writeFile(sourceTextList, targetTextList, outputFolderName)


if __name__ == "__main__":
    folderName = sys.argv[1]
    folderIdentifier = sys.argv[2]
    outputFolderName = sys.argv[3]

    if not os.path.isdir("dataAugT5"):
        os.mkdir("dataAugT5")

    if folderIdentifier == "banking_conf":
        conf = banking77_conf
    elif folderIdentifier == "art_conf":
        conf = art_conf

    seed_everything(40)
    process(folderName, outputFolderName)







