"""
File 5
We train the model using the transformer on input and output vocabulary
"""

import main
import train_val
import vocab
import transformer

import pickle

# Loading the pre-saved vocabulary files from vocab.py
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

enc = transformer.Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, main.device)
dec = transformer.Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, main.device)

print(out_vocab.__dict__['freqs'])

SRC_PAD_IDX = in_vocab.stoi[vocab.Input.pad_token]
TRG_PAD_IDX = out_vocab.stoi[vocab.Output.pad_token]

model = transformer.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, main.device).to(main.device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The model has ", count_params(model), " trainable parameters")