# Essential imports
import torch
from tokenize import untokenize

# User imports
import main
import train_val
import vocab
import transformer
import training
import loss_fun
import eval

# Variables used from user imports
device = main.device
Input = vocab.Input
Output = vocab.Output
model = training.model
translate = eval.translate

model.load_state_dict(torch.load("/Users/vamsi/Desktop/Learning/My Git/GPT_CodeGen/model/model.pt"))            #Check directory path

def eng_to_py(src):
    src = src.split(" ")
    translation, attention = translate(src, SRC, TRG, model, device)
    print(untokenize(translation[:-1]).decode('utf-8'))

SRC = Input
TRG = Output

while True:
    print("\n")
    print("\n")
    src = input("Please give your Python problem statement: \n")
    print("Your requested code: \n")
    eng_to_py(src)