
###################################
########Q1: Consider the following toy data. Imagine this matrix is an embedding matrix of 5 tokens. 
#The embedding matrix has 6 dimensions.
#Calculate the attention output.
import numpy as np
embeddings = np.arange(0,30).reshape(5,6)
embeddings = embeddings / 10.0 #scale to avoid numerical overflow
#Answer:

#alternatively take a matrix of random numbers

def softmax(input):
        def soft(input):
            softmax = np.exp(input) / np.sum(np.exp(input))
            return softmax
        softmax_rowwise = np.apply_along_axis(soft, 1, input)
        return softmax_rowwise

#check if each row adds up to 1
# np.sum(softmax(embeddings),1)

Z = np.matmul(softmax(np.matmul(embeddings,np.transpose(embeddings))/np.sqrt(embeddings.shape[1])),embeddings)

#Z is the attention output
#Z tells us how much should each query (each row), should pay attention to the other rows




###################################
########Q2: Add causal (can only look at previous tokens and the current token) masking to your Q1 solution
#Answer:

def softmax(input):
        def soft(input):
            softmax = np.exp(input) / np.sum(np.exp(input))
            return softmax
        softmax_rowwise = np.apply_along_axis(soft, 1, input)
        return softmax_rowwise

#check if each row adds up to 1
# np.sum(softmax(embeddings),1)
# embeddings
M = np.triu(np.ones(25).reshape(5,5),k=1)
M[M.astype('bool')] = -np.inf
print(M)
# softmax(M)
Z = np.matmul(softmax(np.add(np.matmul(embeddings,np.transpose(embeddings))/np.sqrt(embeddings.shape[1]),M)),embeddings)
print(Z)

###################################
########Q3: Add linear biases to your Q2 solution.
#Answer:

m = 1.0
B = np.array([[0,-np.inf,-np.inf,-np.inf,-np.inf],
              [-1,0,-np.inf,-np.inf,-np.inf],
              [-2,-1,0,-np.inf,-np.inf],
              [-3,-2,-1,0,-np.inf],
              [-4,-3,-2,-1,0]])
B = B * m
Z = np.matmul(softmax(np.add(np.add(np.matmul(embeddings,np.transpose(embeddings))/np.sqrt(embeddings.shape[1]),M),B)),embeddings)
print(Z)
#note: the M matrix can be removed, as the B matrix also masks
