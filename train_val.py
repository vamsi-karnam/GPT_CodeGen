"""
File 2
In this file we build the training and validation datasets.
"""

import main                                                                     #Using the functions we built in main
import numpy as np
import pandas as pd

python_problems_df = pd.DataFrame(main.dps)
#print(python_problems_df.head())
#print(python_problems_df.shape)

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.85                            #Splitting the dataset -- 85% train and 15% validation

train_df = python_problems_df[msk]
val_df = python_problems_df[~msk]

#print(train_df.shape)
#print(val_df.shape)