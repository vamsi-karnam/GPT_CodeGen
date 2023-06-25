"""
File 5
We train the model using the transformer on input and output vocabulary
"""

# Essential imports
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import BucketIterator, Iterator, Field
from torchtext import data

import time
import math
import pickle

from tqdm import tqdm

# User imports
import main
import train_val
import vocab
import transformer

# Variables used from user imports
device = main.device
train_df = train_val.train_df
val_df = train_val.val_df
Input = vocab.Input
Output = vocab.Output
fields = vocab.fields
Encoder = transformer.Encoder
Decoder = transformer.Decoder


# Loading the pre-saved vocabulary files from vocab.py (Check directory before running)
with open("/Users/vamsi/Desktop/Learning/My Git/GPT_CodeGen/vocab/src_vocab.pkl", "rb") as f1:
    in_vocab = pickle.load(f1)

with open("/Users/vamsi/Desktop/Learning/My Git/GPT_CodeGen/vocab/trg_vocab.pkl", "rb") as f2:
    out_vocab = pickle.load(f2)    

INPUT_DIM = len(in_vocab)
OUTPUT_DIM = len(out_vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

#print(out_vocab.__dict__['freqs'])

SRC_PAD_IDX = in_vocab.stoi[Input.pad_token]
TRG_PAD_IDX = out_vocab.stoi[Output.pad_token]

model = transformer.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

# Counting the trainable parameters (~ 5 to 10 million params)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print("The model has ", count_params(model), " trainable parameters")

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim()>1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(init_weights)

LEARNING_RATE = 0.0005
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#print("Trained the seq2seq model on the vocabulary successfully")