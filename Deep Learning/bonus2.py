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
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 100
ct = np.ones(n)

X = np.random.normal(size = n)
X = X / 3.0

ct_X = np.array(np.column_stack((ct, X)))

y = ct * 1.2222 + X * 2.4675 - 8 * X**2
plt.scatter(X, y)
plt.show()

eps = 1e-5

# Weights: Input (2) -> Hidden1 (2)
W_i_h1 = np.array(np.random.normal(size=4)).reshape(2, 2)
# Weights: Hidden1+bias (3) -> Hidden2 (2)
W_h1_h2 = np.array(np.random.normal(size=6)).reshape(3, 2)
# Weights: Hidden2+bias (3) -> Output (1)
W_h2_o = np.array(np.random.normal(size=3)).reshape(3, 1)

def error(W_i_h1, W_h1_h2, W_h2_o, X, y):
    # Hidden layer 1
    z_h1 = np.matmul(X, W_i_h1)
    h1 = 1 / (1 + np.exp(-z_h1))
    # Hidden layer 2
    h1_with_bias = np.column_stack((np.array([1.0]), h1.reshape(1, 2)))
    z_h2 = np.matmul(h1_with_bias, W_h1_h2)
    h2 = 1 / (1 + np.exp(-z_h2))
    # Output layer
    h2_with_bias = np.column_stack((np.array([1.0]), h2.reshape(1, 2)))
    o = np.matmul(h2_with_bias, W_h2_o)
    error = (y - o)**2
    return error.item()

eta = 0.01

for j in range(1000):
    for i in range(len(ct_X)):
        # W_i_h1 gradients (2x2 = 4 weights)
        impact_ih1_00 = (error(np.array([[W_i_h1[0,0]+eps,W_i_h1[0,1]],[W_i_h1[1,0],W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i])-error(np.array([[W_i_h1[0,0]-eps,W_i_h1[0,1]],[W_i_h1[1,0],W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_ih1_01 = (error(np.array([[W_i_h1[0,0],W_i_h1[0,1]+eps],[W_i_h1[1,0],W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i])-error(np.array([[W_i_h1[0,0],W_i_h1[0,1]-eps],[W_i_h1[1,0],W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_ih1_10 = (error(np.array([[W_i_h1[0,0],W_i_h1[0,1]],[W_i_h1[1,0]+eps,W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i])-error(np.array([[W_i_h1[0,0],W_i_h1[0,1]],[W_i_h1[1,0]-eps,W_i_h1[1,1]]]),W_h1_h2,W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_ih1_11 = (error(np.array([[W_i_h1[0,0],W_i_h1[0,1]],[W_i_h1[1,0],W_i_h1[1,1]+eps]]),W_h1_h2,W_h2_o,ct_X[i],y[i])-error(np.array([[W_i_h1[0,0],W_i_h1[0,1]],[W_i_h1[1,0],W_i_h1[1,1]-eps]]),W_h1_h2,W_h2_o,ct_X[i],y[i]))/(2*eps)

        # W_h1_h2 gradients (3x2 = 6 weights)
        impact_h1h2_00 = (error(W_i_h1,np.array([[W_h1_h2[0,0]+eps,W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0]-eps,W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_h1h2_01 = (error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]+eps],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]-eps],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_h1h2_10 = (error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0]+eps,W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0]-eps,W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_h1h2_11 = (error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]+eps],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]-eps],[W_h1_h2[2,0],W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_h1h2_20 = (error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0]+eps,W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0]-eps,W_h1_h2[2,1]]]),W_h2_o,ct_X[i],y[i]))/(2*eps)
        impact_h1h2_21 = (error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]+eps]]),W_h2_o,ct_X[i],y[i])-error(W_i_h1,np.array([[W_h1_h2[0,0],W_h1_h2[0,1]],[W_h1_h2[1,0],W_h1_h2[1,1]],[W_h1_h2[2,0],W_h1_h2[2,1]-eps]]),W_h2_o,ct_X[i],y[i]))/(2*eps)

        # W_h2_o gradients (3x1 = 3 weights)
        impact_h2o_0 = (error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]+eps],[W_h2_o[1,0]],[W_h2_o[2,0]]]),ct_X[i],y[i])-error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]-eps],[W_h2_o[1,0]],[W_h2_o[2,0]]]),ct_X[i],y[i]))/(2*eps)
        impact_h2o_1 = (error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]],[W_h2_o[1,0]+eps],[W_h2_o[2,0]]]),ct_X[i],y[i])-error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]],[W_h2_o[1,0]-eps],[W_h2_o[2,0]]]),ct_X[i],y[i]))/(2*eps)
        impact_h2o_2 = (error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]],[W_h2_o[1,0]],[W_h2_o[2,0]+eps]]),ct_X[i],y[i])-error(W_i_h1,W_h1_h2,np.array([[W_h2_o[0,0]],[W_h2_o[1,0]],[W_h2_o[2,0]-eps]]),ct_X[i],y[i]))/(2*eps)

        # Update W_i_h1
        W_i_h1[0,0] = W_i_h1[0,0] - eta*impact_ih1_00
        W_i_h1[0,1] = W_i_h1[0,1] - eta*impact_ih1_01
        W_i_h1[1,0] = W_i_h1[1,0] - eta*impact_ih1_10
        W_i_h1[1,1] = W_i_h1[1,1] - eta*impact_ih1_11

        # Update W_h1_h2
        W_h1_h2[0,0] = W_h1_h2[0,0] - eta*impact_h1h2_00
        W_h1_h2[0,1] = W_h1_h2[0,1] - eta*impact_h1h2_01
        W_h1_h2[1,0] = W_h1_h2[1,0] - eta*impact_h1h2_10
        W_h1_h2[1,1] = W_h1_h2[1,1] - eta*impact_h1h2_11
        W_h1_h2[2,0] = W_h1_h2[2,0] - eta*impact_h1h2_20
        W_h1_h2[2,1] = W_h1_h2[2,1] - eta*impact_h1h2_21

        # Update W_h2_o
        W_h2_o[0,0] = W_h2_o[0,0] - eta*impact_h2o_0
        W_h2_o[1,0] = W_h2_o[1,0] - eta*impact_h2o_1
        W_h2_o[2,0] = W_h2_o[2,0] - eta*impact_h2o_2

# Plot results
Xplot = np.linspace(min(X), max(X), 100)
yplot = []

for x_val in Xplot:
    x_input = np.array([1.0, x_val])
    # Hidden layer 1
    z_h1 = np.matmul(x_input, W_i_h1)
    h1 = 1 / (1 + np.exp(-z_h1))
    # Hidden layer 2
    h1_with_bias = np.array([1.0, h1[0], h1[1]])
    z_h2 = np.matmul(h1_with_bias, W_h1_h2)
    h2 = 1 / (1 + np.exp(-z_h2))
    # Output
    h2_with_bias = np.array([1.0, h2[0], h2[1]])
    o = np.matmul(h2_with_bias, W_h2_o)
    yplot.append(o.item())

plt.scatter(X, y, label='Data')
plt.plot(Xplot, yplot, 'r-', label='NN Prediction (2 hidden layers)')
plt.legend()
plt.title('Neural Network with 2 Hidden Layers')
plt.show()

# Note: With only 2 neurons per hidden layer and sigmoid activation,
# the network may struggle to capture the full quadratic relationship.
# More epochs or more neurons could improve the fit.
