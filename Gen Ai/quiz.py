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



################################
# Question 3:
# Consider the following corpus: blinking
# Apply the BPE algorithm until you get to the first merge rule.
# What is the first merge rule?
# Answer: 


################################
# Question 4:
# In our discussion about the attention mechanism we compared plain masking using the M matrix,
# and ALiBi using the B matrix.
# How are softmax(M) and softmax(B) different?
# Answer:



################################
# Question 5:
#According to the Chinchilla scaling laws, there exists a compute-optimal relationship between 
#model size (parameters) and dataset size (tokens).
#1.	If a model has 10 billion parameters, how many training tokens should it ideally be trained on?
#2.	If you have a dataset of 2 trillion tokens, how many parameters should a compute-optimal model have?
#3.	Why were many early LLMs suboptimal before Chinchilla?

#Answer:



################################
# Question 6:
#Greedy search and beam search attempt to maximize sequence probability. 
#However, research shows that these methods often produce degenerate (bland or repetitive) text.
#1.	Why is this counterintuitive given that models are trained using maximum likelihood?
#2.	Why do top-k and top-p sampling often produce better text?
#3.	What tradeoff do sampling methods introduce?

#Answer:


################################
# Question 7:
#Why is bias (e.g., treating one group of people differently from another) 
#mitigation fundamentally harder in LLMs than in traditional structured ML systems?

#Answer:


################################
# Question 8:
#True or False, and explain.
#For generative AI, removing punctuation during tokenization is likely to improve model performance.

#Answer: 



################################
# Question 9:
#True or False, and explain.
#Self-attention inherently understands token order.

#Answer: 


################################
# Question 10:
#Give three failure modes for in-context learning.

#Answer:



################################
# Question 11:
# Explain intuitively why dot products measure “attention relevance.”

#Answer:


################################
# Question 12:
# Explain intuitively why subword tokenization is superior to word-based and character-based tokenization.

#Answer:



################################
# Question 13:
#Explain intuitively why RAG reduces hallucinations.

#Answer:



################################
# Question 14:
#Suppose you build a chatbot using:
#Few-shot prompting
#Beam search
#No output verification
#No RAG

#Identify two likely failure modes and explain why.

#Answer:


################################
# Question 15:

#Explain how attention enables the position-wise feed-forward network (FFN) to access prior tokens.

#Answer:


################################
# Question 16:

#You are designing a long-document legal assistant.

#Which three design choices matter most and why?

#Answer:

