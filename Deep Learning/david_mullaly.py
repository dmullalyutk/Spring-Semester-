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
# We used a water treatment plant analogy.
# The raw data entering the network is like water flowing into
# the plant. Each neuron is a basin that applies a
# specific treatment operation to whatever flows into it.
# The weighted connections between neurons are the pipelines
# or tubes that carry the water from one
# basin to the next. By the time the water exits the last basin
#it has been progressively transformed
# into a clean, useful prediction, just as raw water is
# progressively purified into drinking water.

#############################################################
#Question 2

#Consider the following situation. You work in a data science team at a company. You have a large number of records, 1 billion to be precise, and the records are growing in size every day. 
#Your superior has decided that this should be analyzed with a random forest (ensemble of decision trees: a non-linear algorithm) 
#because it can automatically learn non-linear functions and that would benefit predictive performance. 
#Because of time and compute constraints your superior also has decided that the team should use a random subset of data of 1 million records. 
#A larger set would not fit in the server's memory and using a subset would also mean that the very first results of the model would come in on time, 
#before the deadline that the team is currently facing. Convince your superior that deep neural networks are a better choice in this situation. 
#Structure your arguments, and use any theoretical concepts we have seen in class to substantiate your arguments.

#Answer: DROPPING THIS QUESTION


#############################################################                                   
#Question 3
# Which method have we seen in class to visualize the shape of the relationship - as estimated by a deep neural network - between an input and the output. 
# The answer should be no longer than 5 words.
#Answer:
#Partial dependence plot


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

# Model: y_hat = w*x
# Loss for one record: L(w) = (y - w*x)^2
# Analytical derivative: dL/dw = 2*(w*x - y)*x

y_hat = w * x
analytic_grad = 2.0 * (y_hat - y) * x

def mse_loss(w_val):
    return (y - w_val * x) ** 2

numerical_grad = (mse_loss(w + eps) - mse_loss(w - eps)) / (2.0 * eps)

print("Q4 analytical dL/dw:", analytic_grad)
print("Q4 numerical  dL/dw:", numerical_grad)

#The numerical and analytical derivatives are approximately the same 




#############################################################   
#Question 5
#Consider the following data

import numpy as np
import tensorflow as tf
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
inputs = tf.keras.Input(shape=(2,))
hidden = tf.keras.layers.Dense(2, activation='sigmoid')(inputs)
out1   = tf.keras.layers.Dense(1, activation='linear',  name='output1')(hidden)  # continuous
out2   = tf.keras.layers.Dense(1, activation='sigmoid', name='output2')(hidden)  # binary

model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])

model.compile(
    optimizer='adam',
    loss={
        'output1': 'mse',                  # MSE for continuous target
        'output2': 'binary_crossentropy'   # BCE for binary target
    },
    loss_weights={
        'output1': 0.3,
        'output2': 0.7
    }
)

model.fit(
    X,
    {'output1': y1, 'output2': y2},
    batch_size=1,
    epochs=20,
    verbose=0
)

# 2. Prediction on first instance
pred = model.predict(X[:1], verbose=0)
print(f"Prediction – output1 (y1 continuous) : {pred[0][0][0]:.4f}")
print(f"Prediction – output2 (y2 probability): {pred[1][0][0]:.4f}")

# 3. Evaluate on entire dataset
eval_result = model.evaluate(
    X, {'output1': y1, 'output2': y2}, verbose=0
)
print(f"\nEvaluation  [total_loss, out1_loss, out2_loss]: "
      f"{[round(v, 4) for v in eval_result]}")

# 4. Model summary
model.summary()





#####################################################
#Question 6:
# What is the highest value of the derivative of the elu activation function. Alpha=1.
# Create a function called deriv_elu, use a range of input values to determine the maximum, and plot the function.
# Answer:
import matplotlib.pyplot as plt

def deriv_elu(z, alpha=1.0):
    z = np.asarray(z)
    return np.where(z > 0, 1.0, alpha * np.exp(z))

z_values = np.linspace(-8, 8, 2000)
deriv_values = deriv_elu(z_values, alpha=1.0)
max_derivative = float(np.max(deriv_values))

print("Q6 highest derivative of ELU (alpha=1):", max_derivative)
# The highest value of the derivative of the ELU activation function with alpha=1 is 1.0.


plt.figure(figsize=(8, 4))
plt.plot(z_values, deriv_values)
plt.axhline(max_derivative, linestyle="--")
plt.title("Derivative of ELU (alpha=1)")
plt.xlabel("z")
plt.ylabel("dELU/dz")
plt.tight_layout()
plt.show()


####################################################
# Question 7

# Explain gradient descent to a layperson as we did in class.
# Answer:
# Gradient descent is like trying to walk to the bottom of a hill in fog.
# You cannot see the whole landscape, so at each step you feel the local slope,
# then move a little downhill. Repeating that many times usually gets you near
# the lowest point, which means lower error.
#
# In neural network terms:
#   • The landscape = the loss surface (error as a function of
#     all the weights).
#   • Your current position = the current set of weights.
#   • The slope you feel underfoot = the gradient (derivative of
#     the loss with respect to each weight).
#   • The step size = the learning rate.
#
# By repeatedly stepping in the direction opposite to the
# gradient, the model gradually reduces its error, just like
# descending to the bottom of a valley


#############################################################
#Question 8
#Is this statement true or false. Provide reasoning. 
# "Learning in batches of 50 records at a time is 50 times faster than learning in batches of 1 record at a time.""

#Answer:
#Answer:
# False.
# Larger batches reduce some overhead and can use hardware parallelism better,
# but speedup is not linear in batch size. A batch size of 50 performs ~50× fewer updates
# per epoch, but each update is ~50× more expensive, so the
# total computation per epoch is roughly the same.


####################################################
# Question 9

# What are the advantages of having a batch size of 1?
# Answer:
# The biggest advantages of having a batch size of 1 is 
# that it allows the model to update its weights after every single record, which can lead to faster convergence in some cases.
# Additionally it has lower RAM requirements, making it more accessible for training on smaller datasets or with limited computational resources.
