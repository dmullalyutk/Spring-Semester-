# Bonus 4

# Run the tokenizer code from class
# Then run the following code.

# On windows you may need to run
# import os
# os.system('color')

import termcolor
from termcolor import colored
color_list = list(termcolor.COLORS.keys())[1:]*10

def show_tokens(input_tokens,decoder,color_list):
    print('Number of tokens: ',len(input_tokens))
    for i, token in enumerate(input_tokens):
        print(colored(token.replace('</w>',' '),color_list[i]),end='')

show_tokens(input_tokens=input.tokens,
            decoder=tokenizer.decode,
            color_list=color_list)

#You should see the decoded sentence in colors.
#The colors indicate the tokens.
#Change the vocabulary size during instantiation of your trainer from the BpeTrainer class
#and run the whole pipeline, each time printing the colored sentence.
#Do this for vocab_size=1000, vocab_size=2000, vocab_size=10000, vocab_size=30000
#What do you observe? Do you have a recommendation as to which vocabulary size is best 
#out of those four options?
#Answer:

# vocab_size=1000, there are 20 tokens, all single characters
# vocab_size=2000, there are 10 tokens, some tokens representing up to four characters
# vocab_size=10000, there are 6 tokens, some tokens representing up to four characters
# vocab_size=30000, there are 6 tokens, some tokens representing up to four characters

# Vocabulary size 10000 seems sufficient for this particular sentence.