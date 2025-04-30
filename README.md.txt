README: 

In order to properly run the code initialize a terminal and call training_tokenizer.py
or training_base.py then use the parser to set parameters and save files as desired.

The code has been modified to include a new parameter within training_tokenizer: --destiny_file
destiny file needs a csv file route on as a string for input. It records the printed data and 
other parameters of the model within a CSV file that can be accessed to generate different trends and 
curves. The default implementation has './Data_Params/default.csv' as the destiny_file, so in order for 
the code to work you must have a relative file within the repository to have this name. 

With testing_tokenizer.py use the parameter vocab_size as the same vocabulary size used for training the model
you are using as checkpoint to train on. 

Significant modifications:

Modified the BPE tokenization algorithm for training_tokenizer.py, in order to do this I
modified the helper function of utils.py that gets the hashMap with the character pairs and their
frequencies. Added boundary conditions and restrictions to the generation of said map. 

Furthermore, a new counter j was included within the tokenizer file to count for aditional iterations without
adding vocabulary.

Added the CSV file printing algorithm to be a part of the training loop, it verifies if the file exists to 
print headers or not and then opens it in append mode to modify it constantly. Uses the new parser argument 
desiny file to do so.

Added the BPE tokenization algorithm for testing_tokenizer.py it uses a similar outline to the training tokenizer
but it is neccesary in order to encode the text_answer (ground truth) as well as getting the correct vocabulary 
hashMap. The BPE encodes and then decodes into text the generated response from the model to then use the standard performance metrics.

Added the cosine_sim function to utils.py, it uses the encoded versions of the response and the answer (ground truth) to 
access their embedding vectors, get an average of them and then calculate the cosine similarity between the two.