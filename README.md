# GPT_CodeGen

## V1. Python
- An attempt to build a Python code generator based on Transformer architecture.
- The training data has been obtained from a custom dataset available through 'The School of AI' with 5000 datapoints in the format of problem statements and code solutions in Python.

### Notes:
- Considering the dataset is small in size, there is a chance of improper model training, hence the following methods have been used where applicable to avoid problems.
  1. The data can be augmented by changing variable names in the default code to avoid overtraining the code generator on specific variable names.
  2. Additional data can be added or substituted to make the model understand inherent logic and syntax.
  3. Adding different logical or syntactical versions of the same code can improve the models understanding of code.
