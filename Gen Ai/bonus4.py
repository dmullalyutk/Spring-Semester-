# Bonus 4

# Run the tokenizer code from class
# Then run the following code.

# On windows you may need to run
import os
os.system('color')

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def get_corpus():
    for i in range(0, len(dataset), 500):
        yield dataset[i:i+500]['text']

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import decoders

tokenizer = Tokenizer(BPE(unk_token="UNK"))

trainer = BpeTrainer(special_tokens=["UNK", "CLS", "SEP", "PAD", "MASK"],
                     vocab_size=30_000,  # <-- CHANGE THIS VALUE: 1000, 2000, 10000, 30000
                     end_of_word_suffix="</w>")

tokenizer.normalizer = normalizers.Sequence([Lowercase()])
tokenizer.pre_tokenizer = WhitespaceSplit()
tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

tokenizer.train_from_iterator(get_corpus(), trainer=trainer)

input = tokenizer.encode("Lets make some cookies!")

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
# ============== END PROVIDED BONUS CODE ==============

#You should see the decoded sentence in colors.
#The colors indicate the tokens.
#Change the vocabulary size during instantiation of your trainer from the BpeTrainer class
#and run the whole pipeline, each time printing the colored sentence.
#Do this for vocab_size=1000, vocab_size=2000, vocab_size=10000, vocab_size=30000
#What do you observe? Do you have a recommendation as to which vocabulary size is best
#out of those four options?

#Answer:
#
# Results:
#   vocab_size=1000:  20 tokens
#   vocab_size=2000:  11 tokens
#   vocab_size=10000: 7 tokens
#   vocab_size=30000: 7 tokens
#
# Observations:
# - Smaller vocab sizes (1000, 2000) produce MORE tokens because words are broken
#   into smaller subword units. The tokenizer has fewer learned merges.
# - Larger vocab sizes (10000, 30000) produce FEWER tokens because common words
#   are kept whole. The tokenizer has more learned merges.
# - Diminishing returns: Both 10,000 and 30,000 produce 7 tokens, meaning the
#   sentence is fully tokenized at word-level by 10,000.
#
# Recommendation:
# vocab_size=10000 is the best choice because:
#   1. It achieves the same compression as 30,000 (7 tokens)
#   2. Smaller embedding matrix = less memory and faster training
#   3. Good balance between vocabulary coverage and model efficiency
# vocab_size=30000 offers no additional benefit for common text but increases
# model size. vocab_size=1000-2000 fragments words too much, which can hurt
# semantic understanding.
