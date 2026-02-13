####### Question 1
#Consider the following unactivated weighted sum
import numpy as np
unactivated_weighted_sum = np.array([8.7960272 , 8.39056209, 9.30685282, 0.78965963])
#Calculate the softmax with temperature equal to 1.0, and with temperature equal to 100.0.
# Plot both using a barplot.
#Answer:
temp = 1.0
softmax_output = np.exp(unactivated_weighted_sum/temp) / np.sum(np.exp(unactivated_weighted_sum/temp))
np.round(softmax_output,3)

import matplotlib.pyplot as plt
plt.bar(x=[0,1,2,3],height=np.round(softmax_output,3))
plt.title('Temperature=1.0')
plt.ylabel('Softmax output')
plt.xlabel('Unactivated weighted sums')
plt.xticks([0, 1, 2, 3], unactivated_weighted_sum)
plt.show()

temp = 100.0
softmax_output = np.exp(unactivated_weighted_sum/temp) / np.sum(np.exp(unactivated_weighted_sum/temp))
np.round(softmax_output,3)

import matplotlib.pyplot as plt
plt.bar(x=[0,1,2,3],height=np.round(softmax_output,3))
plt.title('Temperature=100.0')
plt.ylabel('Softmax output')
plt.xlabel('Unactivated weighted sums')
plt.xticks([0, 1, 2, 3], unactivated_weighted_sum)
plt.show()


#######Question 2
#Run the following code

from transformers import T5ForConditionalGeneration, T5Tokenizer, TextStreamer, pipeline

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    torch_dtype="auto")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

streamer = TextStreamer(tokenizer)

#Write and run the pipeline for plain vanilla sampling that we saw in class.
#Run it once for temperature=0.5 and once for temperature=100.0.
#How would you describe the quality of the outputs when using a temperature of 100.0? 
#Which one is better, the output with temp=0.5, or the output with temp=100.0.

#Answer:

decoded_inputs = 'How do you make chocolate chip cookies?'
for temperature in [0.5,100.0]:
    generator = pipeline(
        'text2text-generation',
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 50,
        num_beams = 1,
        do_sample = True,
        top_k = 0,
        temperature = temperature,
        device = 'cpu')
    decoded_outputs = generator(decoded_inputs)[0]['generated_text']
    print('TEMP=',temperature,': ',decoded_outputs)

# Temp=100 is gibberish, Temp=0.5 is much better