# bonus 5

###################################
########Q1: Consider the following toy data. Imagine this matrix is an embedding matrix of 5 tokens. 
#The embedding matrix has 6 dimensions.
#Calculate the attention score.
import numpy as np
embeddings = np.arange(0,30).reshape(5,6)
embeddings = embeddings / 10.0 #scale to avoid numerical overflow
#Answer:
d_k = embeddings.shape[1]  # embedding dimension = 6

# Q, K, V are all the same for self-attention
Q = K = V = embeddings

# Attention scores = softmax(Q @ K^T / sqrt(d_k))
scores = Q @ K.T / np.sqrt(d_k)

# Apply softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
attention_output = attention_weights @ V

print("Q1 - Attention Scores (before softmax):")
print(scores)
print("\nQ1 - Attention Weights (after softmax):")
print(attention_weights)


###################################
########Q2: Add causal (can only look at previous tokens and the current token) masking to your Q1 solution
#Answer:
# Create causal mask (lower triangular matrix)
seq_len = embeddings.shape[0]
causal_mask = np.tril(np.ones((seq_len, seq_len)))

# Apply mask: set future positions to -infinity before softmax
scores_masked = scores.copy()
scores_masked[causal_mask == 0] = -np.inf

attention_weights_causal = softmax(scores_masked)
attention_output_causal = attention_weights_causal @ V

print("\nQ2 - Causal Mask:")
print(causal_mask)
print("\nQ2 - Attention Weights (with causal masking):")
print(attention_weights_causal)



###################################
########Q3: Add linear biases to your Q2 solution.
#Answer:
# ALiBi (Attention with Linear Biases)
# Bias = -m * distance, where distance = |i - j|
m = 0.5  # slope parameter

# Create distance matrix
positions = np.arange(seq_len)
distance_matrix = positions[:, None] - positions[None, :]  # i - j

# For causal: only negative distances matter (looking back), set positive to 0
alibi_bias = -m * distance_matrix  # negative values penalize distant tokens
alibi_bias[causal_mask == 0] = -np.inf  # still apply causal mask

# Add bias to scores
scores_with_alibi = scores + alibi_bias

attention_weights_alibi = softmax(scores_with_alibi)
attention_output_alibi = attention_weights_alibi @ V

print("\nQ3 - ALiBi Bias Matrix:")
print(alibi_bias)
print("\nQ3 - Attention Weights (with causal masking + ALiBi):")
print(attention_weights_alibi)