#Quiz 1, Feb 19, 2026
#Deadline: 12:35pm
#Submission link (this is a different link from the usual one):  https://www.dropbox.com/request/KV98BPk8zFSATpYqwvEl
#Filename: firstname_lastname_quiz1.py (please do not submit notebook files)
#Grading:
# You can drop one question by writing: 'DROPPING THIS QUESTION'.

import numpy as np

#############################################################
#Question 1
# Which analogy did we use in class to explain neural networks to laypersons?
# Describe this analogy in your own words.

# Answer:
# A neural network can be explained as a company made of small specialist workers.
# The first workers receive raw information, each next worker combines and transforms it,
# and the final manager gives the decision. During training, feedback tells each worker
# how to adjust slightly so the whole company makes better decisions next time.

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
# 1) Scalability: Deep neural networks can be trained incrementally with mini-batch SGD,
#    so we do not need to load all 1 billion records in memory at once.
# 2) Use all data, not just a small subset: Random forest in this setup would force us to
#    throw away most data (train on 1M only). A neural network can keep learning as new chunks
#    arrive, so signal from the full 1B records is not wasted.
# 3) Better compute efficiency at scale: Neural networks are optimized for vectorized/GPU
#    computation. Training time per record can be much lower when data volume is very large.
# 4) Non-linearity and interactions: Deep networks also learn complex non-linear relationships,
#    and depth helps learn hierarchical feature interactions automatically.
# 5) Future-proofing: Data is growing daily. Neural networks support continuous updates,
#    while repeatedly rebuilding large tree ensembles can become expensive and rigid.
# 6) Practical plan: deliver quick baseline with early epochs, then keep improving by
#    continuing training on new mini-batches before and after the deadline.

#############################################################
#Question 3
# Which method have we seen in class to visualize the shape of the relationship - as estimated by a deep neural network - between an input and the output.
# The answer should be no longer than 5 words.
#Answer:
# Partial dependence plot



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
# The analytical and numerical deriviatives 


#############################################################
#Question 5
#Consider the following data

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
np.random.seed(42)
tf.random.set_seed(42)

y2_float = y2.astype(np.float32)

inputs = tf.keras.layers.Input(shape=(X.shape[1],), name="input")
hidden = tf.keras.layers.Dense(units=2, activation="sigmoid", name="hidden1")(inputs)
output1 = tf.keras.layers.Dense(units=1, activation="linear", name="output1")(hidden)
output2 = tf.keras.layers.Dense(units=1, activation="sigmoid", name="output2")(hidden)

model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss={"output1": "mse", "output2": "binary_crossentropy"},
    loss_weights={"output1": 0.3, "output2": 0.7},
    metrics={"output2": ["accuracy"]}
)

model.fit(
    X,
    {"output1": y1, "output2": y2_float},
    batch_size=1,
    epochs=20,
    verbose=0
)

pred_y1, pred_y2 = model.predict(X[0:1], verbose=0)
print("Q5 first-instance prediction output1 (y1):", float(pred_y1[0, 0]))
print("Q5 first-instance prediction output2 (prob y2=1):", float(pred_y2[0, 0]))

eval_results = model.evaluate(
    X,
    {"output1": y1, "output2": y2_float},
    batch_size=1,
    verbose=0,
    return_dict=True
)
print("Q5 evaluation:", eval_results)

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


#############################################################
#Question 8
#Is this statement true or false. Provide reasoning.
# "Learning in batches of 50 records at a time is 50 times faster than learning in batches of 1 record at a time.""

#Answer:
# False.
# Larger batches reduce some overhead and can use hardware parallelism better,
# but speedup is not linear in batch size. Each update with batch=50 costs more
# computation than batch=1, and convergence behavior can change. In practice,
# batch=50 might be faster wall-clock, slower, or similar depending on hardware,
# implementation, and optimization dynamics.



####################################################
# Question 9

# What are the advantages of having a batch size of 1?
# Answer:
# 1) Very low memory use.
# 2) Can learn online/incrementally as each record arrives.
# 3) Parameter updates happen very frequently, often giving fast early progress.
# 4) Gradient noise can help escape flat regions/shallow local minima.
# 5) Useful when data does not fit in memory.


