from cProfile import label
import sys
import random
import os

# sample script - python3 ssc_scripts/convertBrackToAugClarify.py sscData/ART_split/dev_clean.txt data/ART_SPLIT valid 

def positionOnly(inputFolderName, outputFolderName, fileType):
    dataFile = []
    with open(inputFolderName) as f:
        dataLocal = []
        while True:
            
            line = f.readline()
            if len(line.split("\t")) < 2:
                dataFile.append(dataLocal)
                dataLocal = []
            if not line:
                break
            dataLocal.append(line.split("\t"))
    
    labelList = []
    textList = []
    os.mkdir(os.path.join(outputFolderName,fileType))
    for item in dataFile:
        for index, content in enumerate(item):
            if len(content) == 2:
                labelList.append(content[0])
                textList.append(content[1])
    
    with open(os.path.join(outputFolderName,fileType,"label" ), "w") as f:
        for item in labelList:
            f.write(item + "\n")
    
    with open(os.path.join(outputFolderName,fileType,"seq.in" ), "w") as f:
        for item in textList:
            f.write(item)
    
    print("[done converting dataset - ", outputFolderName, "]")

if __name__ == "__main__":
    inputFolderName = sys.argv[1]
    outputFolderName = sys.argv[2]
    fileType = sys.argv[3]
    positionOnly(inputFolderName, outputFolderName, fileType)