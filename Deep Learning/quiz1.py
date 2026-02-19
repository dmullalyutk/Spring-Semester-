#Quiz 1, Feb 19, 2026
#Deadline: 12:35pm
#Submission link (this is a different link from the usual one):  https://www.dropbox.com/request/KV98BPk8zFSATpYqwvEl
#Filename: firstname_lastname_quiz1.py (please do not submit notebook files)
#Grading: 
	#All questions have equal weight.
	#You can drop one question by writing: 'DROPPING THIS QUESTION', this will make you exempt to the question,
	#as if the question never existed. Each of the remaining questions will count 1/(remaining # of questions) of the total grade.
		#If you do not write this sentence, the question will be graded. 
		#If you use this sentence more than one times the first instance will be used.

#############################################################
#Question 1
# Which analogy did we use in class to explain neural networks to laypersons?
# Describe this analogy in your own words.

# Answer:

#############################################################
#Question 2

#Consider the following situation. You work in a data science team at a company. You have a large number of records, 1 billion to be precise, and the records are growing in size every day. 
#Your superior has decided that this should be analyzed with a random forest (ensemble of decision trees: a non-linear algorithm) 
#because it can automatically learn non-linear functions and that would benefit predictive performance. 
#Because of time and compute constraints your superior also has decided that the team should use a random subset of data of 1 million records. 
#A larger set would not fit in the server's memory and using a subset would also mean that the very first results of the model would come in on time, 
#before the deadline that the team is currently facing. Convince your superior that deep neural networks are a better choice in this situation. 
#Structure your arguments, and use any theoretical concepts we have seen in class to substantiate your arguments.

#Answer:


#############################################################                                   
#Question 3
# Which method have we seen in class to visualize the shape of the relationship - as estimated by a deep neural network - between an input and the output. 
# The answer should be no longer than 5 words.
#Answer:



#############################################################    
#Question 4
#Consider a linear model with one weight w, and one output, trained with MSE as loss.
#It found the following estimate: 
w = 6.666 
#Consider the record:
x = 3.3
y = 10.0
# Note:
eps = 0.0000000001
#Manually compute the derivative of the MSE loss wrt w numerically and analytically for that record.
#Do you find the same answer?

#Answer:


#############################################################   
#Question 5
#Consider the following data

import numpy as np
ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y1 = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2
y2 = ct*3.3332 + X1*4.4766 + X2*10.4572 - 6*X1**2
y2 = np.where(y2 >= np.mean(y2), 1, 0)

#y1 is continuous and y2 is binary
#1. Estimate a neural network with one hidden layer (2 units) and two output layers. 
# output1 predicts y1 and output2 predicts y2. Make sure to have the correct activation functions.
# Make sure to have the correct loss functions and assign a weight of 0.3 to the loss
# for output1 and 0.7 to the loss of output2. Batch size = 1, epochs = 20
#2. Make a prediction on the first instance of the dataset
#3. Evaluate the model on the entire dataset
#4. Summarize the model

#Make sure to use tensorflow.

#Answer:


#####################################################
#Question 6:
# What is the highest value of the derivative of the elu activation function. Alpha=1.
# Create a function called deriv_elu, use a range of input values to determine the maximum, and plot the function.
# Answer:



####################################################
# Question 7

# Explain gradient descent to a layperson as we did in class.
# Answer:


#############################################################
#Question 8
#Is this statement true or false. Provide reasoning. 
# "Learning in batches of 50 records at a time is 50 times faster than learning in batches of 1 record at a time.""

#Answer:



####################################################
# Question 9

# What are the advantages of having a batch size of 1?
# Answer:


