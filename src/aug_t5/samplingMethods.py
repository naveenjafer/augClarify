import os
import json
import random

def random_sampling(text,dropProb):
    space_tokenized_text = text.split(" ")
    random.shuffle(space_tokenized_text)

    # drop with a probability of
    space_tokenized_text = random.sample(space_tokenized_text,int(len(space_tokenized_text)*dropProb))
    return space_tokenized_text
