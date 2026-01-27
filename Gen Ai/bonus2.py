
# Bonus 2:

# Q1:
# Install all the required software, load google/flan-t5-base, and prompt the LLM with ''How do you play soccer?''
# Paste the answer below.
# Answer: Play with your feet on the ground.

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Q1 prompt
prompt = "How do you play soccer?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Q1 Prompt: {prompt}")
print(f"Q1 Answer: {answer}")


# Q2:
# We expect the flan-t5 model to be substantially worse than ChatGPT because it is much smaller.
# Find one prompt where flan-t5 is substantially worse than ChatGPT (accessible at https://chatgpt.com/)
# Find one prompt where flan-t5 and ChatGPT provide answers of similar quality.
# Paste your two prompts and the answers from flan-t5 and ChatGPT below.
# Please remember that the TA and the instructor will read what you write here below. 
# Please use your judgment and only paste text below that would be acceptable in professional settings.
# Answer:

# Example of prompt where flan-t5 is much worse:
# Prompt: "Write a short poem about the feeling of nostalgia on a rainy day"

# flan-t5 answer: i woke up in the morning with a sigh of relief i had a dream i was going to see a movie i had a dream i was going to see a movie i had a dream i was going to see a movie... (repeats endlessly)

# ChatGPT answer:
# Rain taps the glass like it knows my name,
# each drop a soft reminder of somewhere else.
# The sky wears yesterday's colors,
# and the street smells like old summers rinsed clean.
#
# I don't miss anything exactly—
# just the way time once felt slower,
# when storms meant waiting, not rushing,
# and the world paused long enough
# for memory to feel warm.


# Example of prompt where flan-t5 has a similar quality response as ChatGPT:
# Prompt: "Classify the sentiment of this sentence as positive or negative: I love this movie!"

# flan-t5 answer: positive

# ChatGPT answer: Positive
