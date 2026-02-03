from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split = "train")

def get_corpus():
    for i in range(0,len(dataset),500):
        yield dataset[i:i+500]['text']


next(get_corpus())

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="UNK"))

trainer = BpeTrainer(special_tokens=["UNK", "CLS", "SEP", "PAD", "MASK"],
                     vocab_size=30_000,
                     end_of_word_suffix="</w>")

from tokenizers import normalizers
from tokenizers.normalizers import Lowercase

tokenizer.normalizer = normalizers.Sequence([Lowercase()])

from tokenizers.pre_tokenizers import WhitespaceSplit
tokenizer.pre_tokenizer = WhitespaceSplit()

from tokenizers import decoders 
tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

tokenizer.train_from_iterator(get_corpus(), trainer=trainer)

input = tokenizer.encode("Lets make some cookies!")
print(input.tokens)
print(input.ids)
output = tokenizer.decode(input.ids)
print(output)
