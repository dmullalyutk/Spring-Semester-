#Bonus 2

#########################
# Question 1:
# Start from the incremental implementation of a linear model NeuralNetFromScratch.py.
# Plot the evolution of the two parameters across the 1000 epochs. 
# In each epoch, plot the weights after the last update in the epoch 
# (i.e., right before moving on to the next epoch). There should be 1000 values.

#Answer:
import numpy as np
import matplotlib.pyplot as plt

ct = np.ones(20)
x = np.random.normal(size = 20)
ct_x = np.column_stack((ct, x))
y = ct * 2.22 + x * 5.4675

eps = 1e-5
W = np.array(np.random.normal(size =2 )).reshape(2,1)

def error ( weight,x,y):
    y_hat = np.matmul(x,weight).flatten()
    error = (y-y_hat)**2
    return error

eta = .01

# Track weights after each epoch
w0_history = []
w1_history = []

for j in range(1000):
    for i in range(len(ct_x)):
        impact1 = (error(np.array([W[0]+eps,W[1]]),ct_x[i],y[i])-error(np.array([W[0]-eps,W[1]]),ct_x[i],y[i]))/(2*eps)
        impact2 = (error(np.array([W[0],W[1]+eps]),ct_x[i],y[i])-error(np.array([W[0],W[1]-eps]),ct_x[i],y[i]))/(2*eps)

        W[0] = W[0] -  eta*impact1
        W[1] = W[1] - eta* impact2

    # Record weights at end of each epoch
    w0_history.append(W[0][0])
    w1_history.append(W[1][0])

# Plot the evolution of parameters
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1000), w0_history)
plt.xlabel('Epoch')
plt.ylabel('W[0]')
plt.title('Weight 0 (Intercept) Evolution')

plt.subplot(1, 2, 2)
plt.plot(range(1000), w1_history)
plt.xlabel('Epoch')
plt.ylabel('W[1]')
plt.title('Weight 1 (Slope) Evolution')

plt.tight_layout()
plt.show()

#########################
#Question 2: 
# Start from the neural network code with one hidden layer code in NeuralNetFromScratch.py.
#1.1 Extend that code to a network with two hidden layers (each having two hidden neurons and a bias). 
#Use the sigmoid activation function. 
#1.2 Plot the original data and the predicted data. Does it find the quadratic relationship?

#Answer:


