######################################################################

# Quiz, Feb 12, 2025, deadline: 2:10pm
# *****Submission Link:
# https://www.dropbox.com/request/cgqmWMb5lzclvUWwK0qn
# *****Name your file firstname_lastname.py***********
#Make sure you upload the correct file!

######################################################################

# Quiz rules:

# Please refrain from:
#    any form of collaboration or communication with humans (including chat, or social media)

# You are welcome to
#    ask the instructor any questions you may have,
#    use all material provided in class (book, solutions, notes),
#    use the internet (note: no direct or indirect or delayed communication),
#    use AI.

# Late submissions will incur a progressive penalty as detailed in the syllabus.

# Only submit your own work.

######################################################################

# Only provide code and comments, do not copy paste output.

# Topics that are included in this quiz:
# Introduction section
# Transformers section
# Prompt engineering section: first session
# All bonus materials.
# Everything discussed.

# Sessions:
# 1-7


######################################################################
# Grading:

# Each question has equal weight. You can drop one question.
# Indicate the question that you want to drop by writing this sentence as the answer:

# DROPPING THIS QUESTION

# In other words, if you run out of time, you can drop a question without being penalized.
# If you do have the time, then it might be better to answer the question if you think you will get it right,
# as you are spreading the risk over a greater number of questions.

# Note: if DROPPING THIS QUESTION is provided as an answer, the question will not be graded.
# If that sentence is not provided, then the answer will be graded.
# If you use more than one drop, then the first one will be used, and any other questions marked as dropped will be graded.

######################################################################

# Once you have submitted you are free to leave the room, but once you leave the room
# you are not allowed to make additional submissions, even if they are before the deadline.

######################################################################

################################
# Question 1
# A major breakthrough in the field of neural networks was the insight that
# models will generalize better if we make the models deeper, and that they
# will memorize better if we make them wider. With the advent of transformers,
# we realized another major breakthrough. What was this breakthrough
# and how is it different from the other two breakthroughs.
# Answer:

# The third major breakthrough was the self-attention mechanism introduced by transformers.
# Attention is a mechanism for pulling data from other rows (tokens) and aggregating it
# into the current row. Each token can attend to every other token, enabling the model
# to dynamically learn relationships between all tokens in a sequence.
#
# How it differs from the other two:
# - Depth improves generalization by learning hierarchical representations (static architecture).
# - Width improves memorization by increasing capacity (static architecture).
# - Attention creates dynamic, input-dependent connections: the model learns which tokens
#   are relevant to each other and rewrites each token as a mixture of the other tokens.
#   This is fundamentally different because depth and width are fixed structural properties,
#   while attention adapts to the specific input sequence.


################################
# Question 2
################################
# Question 2
import numpy as np
# Consider the watermarking algorithm on slide 33 of the introduction.
# Consider the following prompt: "May the force be with"
# Assume tokens are words.
# Imagine I have run the LLM for you. This completes step 1 of the watermarking algorithm:
# The next-word probability distribution is:
p = {'you':0.7, 'me':0.1 , 'him': 0.15, 'her': 0.05}

# Implement step 2 of the watermarking algorithm.
    # Hint 1: you can use hash(): use the first six integers (otherwise it will be too long of a number)
    # Hint 2: np.random.seed
# Then, implement step 3.
    # Hint 1: np.random.rand()
    # Hint 2: np.argsort()
# Then, implement step 4 using deterministic sampling (i.e., greedy)

# What is the selected word that comes after "May the force be with"?

# Answer:

# Step 2: Compute hash of previous token and use it to seed RNG
prev_token = "with"
h = hash(prev_token)
seed = int(str(abs(h))[:6])  # first six digits
np.random.seed(seed)

# Step 3: Randomly partition vocabulary into green list G and red list R of equal size
tokens = list(p.keys())
probs = np.array(list(p.values()))

# Generate random numbers for each token and argsort to get random ordering
r = np.random.rand(len(tokens))
order = np.argsort(r)

# Green list = first half, Red list = second half
mid = len(tokens) // 2
green_indices = order[:mid]
red_indices = order[mid:]

green_list = [tokens[i] for i in green_indices]
red_list = [tokens[i] for i in red_indices]

print(f"Seed: {seed}")
print(f"Random values: {r}")
print(f"Order: {order}")
print(f"Green list: {green_list}")
print(f"Red list: {red_list}")

# Step 4: Greedy sampling from green list ONLY (hard red list - never generate red tokens)
# Filter to only green list tokens and their probabilities
green_probs = {tokens[i]: probs[i] for i in green_indices}
print(f"Green list probabilities: {green_probs}")

# Greedy: select the token with highest probability from the green list
selected_word = max(green_probs, key=green_probs.get)
print(f"Selected word: {selected_word}")




################################
# Question 3:
# Consider the following corpus: blinking
# Apply the BPE algorithm until you get to the first merge rule.
# What is the first merge rule?
# Answer:

# Starting vocabulary (characters): b, l, i, n, k, i, n, g
#
# Count all adjacent character pairs:
# (b, l): 1
# (l, i): 1
# (i, n): 2   <-- most frequent pair
# (n, k): 1
# (k, i): 1
# (n, g): 1
#
# The first merge rule is: i + n -> in
# because (i, n) is the most frequent pair with count 2.


################################
# Question 4:
# In our discussion about the attention mechanism we compared plain masking using the M matrix,
# and ALiBi using the B matrix.
# How are softmax(M) and softmax(B) different?
# Answer:

# softmax(M) produces uniform attention weights over all allowed (non-masked) positions.
# The M matrix has 0s for allowed positions and -inf for future positions. After softmax,
# each token pays equal attention to all previous tokens and itself.
# For example, token 5 (fifth row) pays 0.2 attention to itself, 0.2 to the token
# immediately preceding it, 0.2 to the token before that, etc.
#
# softmax(B) produces recency-biased attention weights. The B matrix (ALiBi) has 0 on
# the diagonal, -1 for one step back, -2 for two steps back, etc., plus -inf for future
# positions. After softmax, the token pays most attention to itself, less to the
# immediately preceding token, even less to the token before that.
# For example, token 5 (fifth row) pays 0.63 attention to itself, 0.23 to the preceding
# token, 0.08 to the token before that, and so on with decreasing weights.
#
# In summary: softmax(M) treats all past tokens equally (uniform), while softmax(B)
# gives most weight to nearby tokens with linearly decaying bias for distant tokens.
# The M matrix can be removed when using B, as B already masks future tokens with -inf.


################################
# Question 5:
#According to the Chinchilla scaling laws, there exists a compute-optimal relationship between
#model size (parameters) and dataset size (tokens).
#1.	If a model has 10 billion parameters, how many training tokens should it ideally be trained on?
#2.	If you have a dataset of 2 trillion tokens, how many parameters should a compute-optimal model have?
#3.	Why were many early LLMs suboptimal before Chinchilla?

#Answer:

# 1. The optimal training dataset size is 20 times the number of parameters.
#    10 billion parameters * 20 = 200 billion training tokens.
#
# 2. The optimal number of parameters is 5% of the training dataset size.
#    2 trillion tokens / 20 = 100 billion parameters.
#
# 3. Before Chinchilla, many LLMs had too many parameters for the dataset size they
#    were trained on. They were over-parameterized and under-trained. Chinchilla showed
#    that a smaller model trained on more data can outperform a larger model trained on
#    less data, given the same compute budget.


################################
# Question 6:
#Greedy search and beam search attempt to maximize sequence probability.
#However, research shows that these methods often produce degenerate (bland or repetitive) text.
#1.	Why is this counterintuitive given that models are trained using maximum likelihood?
#2.	Why do top-k and top-p sampling often produce better text?
#3.	What tradeoff do sampling methods introduce?

#Answer:

# 1. It is counterintuitive because the model is trained to maximize likelihood, so
#    you would expect that always choosing the highest probability sequence would
#    reconstruct human-like text. However, maximization-based methods like beam search
#    lead to degenerate text (bland, incoherent, repetitive). Human language is not
#    deterministic, humans use varied, sometimes surprising word choices, not always
#    the most probable ones.
#
# 2. Top-k and top-p sampling produce higher quality text. Top-k retains only the k
#    most likely words, avoiding the long tail of low probability words. Top-p selects
#    the smallest set of words whose cumulative probability reaches p. Both introduce
#    controlled randomness while excluding nonsensical outputs.
#
# 3. Sampling introduces a tradeoff between coherence/factuality and diversity/creativity.
#    Higher randomness produces more varied text but risks incoherent outputs. Lower
#    randomness is more predictable but can be bland. Even top-k and top-p sampling
#    have been shown to generate repetitive text in some cases.


################################
# Question 7:
#Why is bias (e.g., treating one group of people differently from another)
#mitigation fundamentally harder in LLMs than in traditional structured ML systems?

#Answer:

# In traditional structured ML, bias mitigation is straightforward. For example, for a
# credit card approval model, we could simply remove gender from the inputs, calibrate
# the model for gender, and audit the model to ensure that the error rate is the same
# for male and female applicants. Inputs and outputs are structured and controllable.
#
# In LLMs, bias mitigation is fundamentally harder because:
# 1. Outputs are open-ended text, not structured predictions. Ensuring fairness for all
#    possible completions (e.g., pronoun assignment for different professions) can quickly
#    become intractable.
# 2. There are three intervention points: filter training data, filter user prompts, or
#    filter generated output. But filtering turns out to be very hard, except when bias
#    is obvious.
# 3. Building guardrail models requires human-annotated data where humans read text and
#    decide whether it is biased. Building consensus on the definition of bias is even
#    harder.
# 4. The model needs to remain consistent with context, e.g., if a prompt mentions long
#    hair, should completions use female pronouns more? This nuance doesn't exist in
#    traditional ML with structured features.


################################
# Question 8:
#True or False, and explain.
#For generative AI, removing punctuation during tokenization is likely to improve model performance.

#Answer:

# False. In traditional text processing pipelines (e.g., text mining), removal of
# punctuation was common because those methods only consider the presence or absence
# of words (bag of words), not their order. Since punctuation appears in all sentences,
# it adds no value in that setting.
#
# However, in modern generative AI pipelines, removing punctuation would remove valuable
# data from the input, as the order of words is important. Punctuation carries
# semantic/syntactic meaning (question marks, periods, commas). The model must produce
# natural text including punctuation, so removing it from training data would degrade
# performance.


################################
# Question 9:
#True or False, and explain.
#Self-attention inherently understands token order.

#Answer:

# False. Self-attention has no built-in notion of token order or distance. The attention
# mechanism computes pairwise dot products between queries and keys regardless of their
# position in the sequence. Without positional information, self-attention treats the
# input as a set, not a sequence.
#
# This is why positional encodings are essential. Previous versions of the transformer
# used positional embeddings added to the token embeddings. A simpler, more efficient,
# and better performing option is ALiBi (Attention with Linear Biases), which adds
# position-dependent linear biases directly into the attention scores.


################################
# Question 10:
#Give three failure modes for in-context learning.

#Answer:

# 1. Wrong labels in examples can throw off the model. For instance, providing an
#    incorrect label (e.g., labeling "This is a great cookie" as "negative") causes
#    the model to produce wrong outputs. The model mimics the pattern of the examples,
#    even if the labels are incorrect.
#
# 2. Sensitivity to example ordering, the order in which few-shot examples are presented
#    can drastically change the model's output. Simply reordering the same examples may
#    lead to very different (and sometimes incorrect) predictions.
#
# 3. Context window limitations, with multiple examples we can quickly surpass the
#    context window length, causing information to be lost. Models with small context
#    windows (e.g., FLAN-T5 with 512 tokens) are especially vulnerable.


################################
# Question 11:
# Explain intuitively why dot products measure "attention relevance."

#Answer:

# The dot product between Q_i and K_j tells us how much query i should pay attention
# to key j (the attention score). Intuitively, if two paired numbers being multiplied
# are both positive or both negative, the product is positive and increases the final
# sum. If they have opposite signs, the product is negative and decreases the sum.
# So when the signs of the paired numbers are aligned, we get a larger sum.
#
# The behavior we want:
# - High attention score for tokens relevant to each other -> vectors are sign-aligned
# - Low attention score for unrelated tokens -> vectors are sign-misaligned
#
# The model learns the linear layer weights so that relevant token pairs produce aligned
# Q and K vectors (high dot product) and irrelevant pairs produce misaligned vectors
# (low dot product). This is why dot products naturally measure attention relevance.


################################
# Question 12:
# Explain intuitively why subword tokenization is superior to word-based and character-based tokenization.

#Answer:

# Word-based: large vocabularies (500,000+ words in English), slow training/inference.
# Words like "cat" and "cats" are separate tokens with no shared meaning initially.
# Unknown words fall back to a shared UNK token, so the model cannot differentiate
# between them at all.
#
# Character-based: very small vocabulary and far fewer unknown tokens. However,
# characters capture less meaning and sequences become very long.
#
# Subword tokenization (e.g., BPE) balances the size of vocabularies, the number of
# out-of-vocabulary tokens, and the length of sequences. Frequently used words stay
# whole; rare words are split into meaningful subparts (e.g., "promotion" -> "promot"
# + "ion"). This is the dominant tokenization method.


################################
# Question 13:
#Explain intuitively why RAG reduces hallucinations.

#Answer:

# RAG (Retrieval-Augmented Generation) reduces hallucinations by grounding the model's
# responses in actual retrieved documents rather than relying solely on parametric memory
# (what the model memorized during training).
#
# Without RAG, when an LLM doesn't know an answer, it may hallucinate, generating
# plausible-sounding but factually incorrect text based on patterns in its training data.
#
# With RAG, the model first retrieves relevant documents from an external knowledge base
# using semantic search (searching by meaning, not keyword matching), then generates its
# response conditioned on that retrieved context. This works because:
# 1. The model can directly reference the retrieved text instead of generating from memory.
# 2. The retrieved context constrains the generation space, reducing unsupported claims.
# 3. The knowledge base can be updated without retraining the model.


################################
# Question 14:
#Suppose you build a chatbot using:
#Few-shot prompting
#Beam search
#No output verification
#No RAG

#Identify two likely failure modes and explain why.

#Answer:

# 1. Hallucination: Without RAG, the chatbot has no external knowledge source to ground
#    its responses. Without output verification, there is no mechanism to catch factual
#    errors. The model will rely entirely on its parametric memory and may confidently
#    generate plausible-sounding but incorrect information, especially for domain-specific
#    or recent topics not well-represented in training data.
#
# 2. Repetitive/bland responses: Beam search maximizes sequence probability, which tends
#    to produce degenerate, repetitive, and generic text. The chatbot will likely give
#    safe but unnatural responses, often repeating phrases or producing overly generic
#    answers rather than engaging, diverse conversation. A sampling method (top-k or
#    top-p) would produce more natural dialogue.


################################
# Question 15:

#Explain how attention enables the position-wise feed-forward network (FFN) to access prior tokens.

#Answer:

# The position-wise FFN applies the feed-forward neural network to each time step (token)
# separately and identically. It has no mechanism to look at other positions on its own.
#
# So how can the FFN access data from previous time steps? The answer is attention.
# Attention constructs a new variable per each input variable that contains data from
# the other time steps (tokens) that were used as input.
#
# Attention precedes the FFN and mixes information across all positions. Each token's
# representation becomes a weighted sum of all allowed tokens' values. By the time the
# FFN receives a token's representation, it already contains aggregated information from
# prior tokens via attention. The FFN then transforms this context-enriched representation.


################################
# Question 16:

#You are designing a long-document legal assistant.

#Which three design choices matter most and why?

#Answer:

# 1. RAG (Retrieval-Augmented Generation): Legal accuracy is critical, incorrect legal
#    advice can have serious consequences. RAG allows the assistant to retrieve relevant
#    statutes, case law, and document sections, grounding its responses in actual legal
#    text rather than relying on potentially outdated or incorrect parametric memory.
#    This significantly reduces hallucination risk in a domain where accuracy is paramount.
#
# 2. Long-context handling strategy: Legal documents (contracts, briefs, regulations) can
#    be extremely long, often exceeding standard context windows. The design must address
#    this through chunking with overlap, hierarchical summarization, or using models with
#    extended context windows. Positional encoding choice matters here too, ALiBi
#    generalizes better to longer sequences than fixed positional encodings.
#
# 3. Output verification / guardrails: Given the high-stakes nature of legal work, the
#    system needs output verification to catch hallucinations, check citations against
#    actual sources, and flag uncertain responses. Without verification, the assistant
#    could generate fabricated case citations or misstate legal precedents, which is
#    unacceptable in a professional legal context.
