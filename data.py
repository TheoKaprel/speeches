import torch
import numpy as np
from functions import create_dicts,encode, decode, create_data
if torch.cuda.is_available():
    dev = "cuda:0"
else:
	dev = "cpu"

# dev = "cpu"
device = torch.device(dev)

# Path to the training data file containing speechs
PATH = 'candidats.txt'

# Text-cutting params
SEQ_LEN = 50
SEQ_SKIP = 3

TEXT = open(PATH).read().lower()[0:1500000]


ALPHABET = sorted(list(set(TEXT)))

NB_CHARS = len(ALPHABET)


CHAR_INDEX, INDEX_CHAR = create_dicts(ALPHABET)

ENCODED_TEXT = encode(TEXT, CHAR_INDEX, NB_CHARS, bool)

INPUTS, TARGETS = create_data(ENCODED_TEXT, SEQ_LEN, SEQ_SKIP)

inputs = torch.from_numpy(INPUTS).float().to(device)
targets = torch.from_numpy(TARGETS).float().to(device)







