# GPT_CodeGen

## V1. Python
- An attempt to build a Python code generator based on Transformer architecture.
- The training data has been obtained from a custom dataset available through 'The School of AI' with 5000 datapoints in the format of problem statements and code solutions in Python.

### Notes:
- Considering the dataset is small in size, there is a chance of improper model training, hence the following methods have been used where applicable to avoid problems.
  1. The data can be augmented by changing variable names in the default code to avoid overtraining the code generator on specific variable names.
  2. Additional data can be added or substituted to make the model understand inherent logic and syntax.
  3. Adding different logical or syntactical versions of the same code can improve the models understanding of code.

### Instructions:
- To train and run the model on your own machine:
1. Clone the repo to your directory.
2. Run the Python files in the following order:
    - First run main.py.
    - Next, run the train_val.py.
    - Next, Uncomment the save functionality in vocab.py and run the vocab.py file with the correct path for vocab files.
    - Run the training.py file to train the model.
    - Change the batch size, number of epochs, and directory as necessary and run model.py.
3. After successfully completing the steps you should find the model saved in the "model" directory.