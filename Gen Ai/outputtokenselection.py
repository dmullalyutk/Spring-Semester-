#####Output token selection

from transformers import T5ForConditionalGeneration, T5Tokenizer, TextStreamer, pipeline

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    torch_dtype="auto")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

streamer = TextStreamer(tokenizer)


#######################################################
#######################################################
####Greedy Search
#######################################################
#######################################################
#We have been doing greedy search all along.
#Greedy search requires 
# num_beams = 1
# do_sample == False

generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    streamer=streamer,
    num_beams = 1,
    do_sample = False,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)



#######################################################
#######################################################
####Beam search
#######################################################
#######################################################
#Note: streamer cannot be used with beam search.
#Beam search requires:
# num_beams > 1
# do_sample = False

generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    do_sample = False,
    num_beams = 2,
    early_stopping = True,
    no_repeat_ngram_size = 2,
    device = 'cpu')
#early_stopping (bool or str, optional, defaults to False) 
# — Controls the stopping condition for beam-based methods, like beam-search. 
# It accepts the following values: 
# True, where the generation stops as soon as there are num_beams complete candidates; 
# False, where an heuristic is applied and the generation stops when is it very unlikely to 
# find better candidates; 
# "never", where the beam search procedure only stops when there cannot be better 
# candidates (canonical beam search algorithm).

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)

#We can request to see all the beams with the num_return_sequences parameter.
# Make sure to set num_return_sequences <= num_beams.

generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    do_sample = False,
    num_beams = 2,
    early_stopping = True,
    no_repeat_ngram_size = 2,
    num_return_sequences = 2,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)
for i,beam_output in enumerate(decoded_outputs):
    print('BEAM ',i,': ', beam_output['generated_text'])


#######################################################
#######################################################
####Plain vanilla Sampling
#######################################################
#######################################################
#Plain vanilla sampling requires
#num_beams = 1
#do_sample = True
#top_k = 0


generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    streamer=streamer,
    num_beams = 1,
    do_sample = True,
    top_k = 0,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)


#temperature

generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    streamer=streamer,
    num_beams = 1,
    do_sample = True,
    top_k = 0,
    temperature = 0.5,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)




#######################################################
#######################################################
####Top-k Sampling
#######################################################
#######################################################
#Top-k sampling requires
#num_beams = 1
#do_sample = True
#top_k > 0


generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    streamer=streamer,
    num_beams = 1,
    do_sample = True,
    top_k = 50,
    temperature = 0.5,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)


#######################################################
#######################################################
####Top-p Sampling
#######################################################
########################################################
# Top-p sampling requires
#num_beams = 1
#do_sample = True
#top_k = 0
# 0 < top_p < 1


generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    streamer=streamer,
    num_beams = 1,
    do_sample = True,
    top_k = 0,
    temperature = 0.5,
    top_p = 0.9,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)[0]['generated_text']
print(decoded_outputs)

#top-p and top-k can be combined 
#when used together, top-k is applied first, and then top-p
#this allows us to filter out some very low probability tokens
#let's also use the num_return_sequences to return multiple text generations so we can observe
#the result of all our hard work!

generator = pipeline(
    'text2text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 50,
    num_beams = 1,
    do_sample = True,
    top_k = 50,
    temperature = 0.5,
    top_p = 0.9,
    num_return_sequences = 3,
    device = 'cpu')

decoded_inputs = 'How do you make chocolate chip cookies?'
decoded_outputs = generator(decoded_inputs)
for i,return_sequence in enumerate(decoded_outputs):
    print('Return sequence ',i,': ', return_sequence['generated_text'])


