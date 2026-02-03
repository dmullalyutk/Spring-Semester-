####### Question 1
#Consider the following unactivated weighted sum
import numpy as np
import matplotlib.pyplot as plt

unactivated_weighted_sum = np.array([8.7960272 , 8.39056209, 9.30685282, 0.78965963])
#Calculate the softmax with temperature equal to 1.0, and with temperature equal to 100.0.
# Plot both using a barplot.
#Answer:

def softmax_with_temperature(x, temperature=1.0):
    """Calculate softmax with temperature scaling."""
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x)

# Calculate softmax with both temperatures
softmax_temp_1 = softmax_with_temperature(unactivated_weighted_sum, temperature=1.0)
softmax_temp_100 = softmax_with_temperature(unactivated_weighted_sum, temperature=100.0)

print("Softmax with temperature=1.0:", softmax_temp_1)
print("Softmax with temperature=100.0:", softmax_temp_100)

# Plot both using bar plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

axes[0].bar(x_labels, softmax_temp_1, color='steelblue')
axes[0].set_title('Softmax with Temperature = 1.0')
axes[0].set_ylabel('Probability')
axes[0].set_ylim(0, 1)

axes[1].bar(x_labels, softmax_temp_100, color='coral')
axes[1].set_title('Softmax with Temperature = 100.0')
axes[1].set_ylabel('Probability')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('Gen Ai/softmax_temperature_comparison.png')
plt.show()


# Observation: Temperature=1.0 gives a peaked distribution (Class 2 dominates)
# Temperature=100.0 gives a nearly uniform distribution (all classes ~0.25)



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

prompt = "Explain how to ride a bike"

# Pipeline with temperature=0.5 (Plain Vanilla Sampling)
pipe_low_temp = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.5,
    max_new_tokens=50,
    device="cpu"
)

# Pipeline with temperature=100.0 (Plain Vanilla Sampling)
pipe_high_temp = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=100.0,
    max_new_tokens=50,
    device="cpu"
)

# Generate with temperature=0.5

print("Temperature = 0.5 (Plain Vanilla Sampling)")


output_low_temp = pipe_low_temp(prompt)
print(f"Output: {output_low_temp[0]['generated_text']}")

# Generate with temperature=100.0
print("\n" + "=" * 50)
print("Temperature = 100.0 (Plain Vanilla Sampling)")
print("=" * 50)

output_high_temp = pipe_high_temp(prompt)
print(f"Output: {output_high_temp[0]['generated_text']}")

# Analysis:
# Temperature = 0.5: Produces more coherent, sensible output. The model is more confident
# and picks high-probability tokens, resulting in grammatically correct text.
#
# Temperature = 100.0: Produces nonsensical, garbled output. The extremely high
# temperature flattens the probability distribution (as seen in Q1), making the model
# sample nearly uniformly across all tokens - essentially random gibberish.
#
# Conclusion: Temperature 0.5 is MUCH better. High temperature (100.0) destroys
# the model's ability to produce meaningful text because it removes the distinction
# between likely and unlikely tokens.
