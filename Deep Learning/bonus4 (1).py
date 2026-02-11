# Bonus 4

####################
# Question 1: 
# Run the following code:
#note: assume n = 1, and a linear model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#loss plot for two artibrarily chosen y-values and an x-value
#and different values of theta           
theta = np.arange(-30,41,dtype=np.float64)  
x = 1
yhat = theta * x      


#Implement the Huber loss manually. Name the function huber_fn and give two arguments: y_true and y_pred. Use tensorflow functions
#tf.abs, tf.square. Then plot the Huber Loss for y=10 and y=0 as per the slides. Take any value for delta (e.g., 5).

#Answer:
delta = 5

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * tf.abs(error) - 0.5 * delta**2
    return tf.where(is_small_error, squared_loss, linear_loss)

# Plot Huber Loss for y=10 and y=0
y_10 = 10
y_0 = 0

loss_y10 = huber_fn(y_10, yhat)
loss_y0 = huber_fn(y_0, yhat)

plt.figure(figsize=(10, 6))
plt.plot(theta, loss_y10, label='y = 10')
plt.plot(theta, loss_y0, label='y = 0')
plt.xlabel('theta')
plt.ylabel('Huber Loss')
plt.title('Huber Loss for y=10 and y=0')
plt.legend()
plt.grid(True)
plt.show()

     


####################
#Question 2:
#Run the following code:
#note: assume n = 1, and a linear model
import matplotlib.pyplot as plt
import numpy as np
theta = np.arange(-30,41,dtype=np.float64)  
x = 1
yhat = theta * x      
y = 0 #true value is arbitrarily chosen to be 0      

#Compute the gradient for the Huber Loss manually and plot the result as in the slides.
#Answer:
delta = 5

def huber_gradient(y_true, y_pred):
    error = y_pred - y_true  # Note: gradient w.r.t. y_pred
    is_small_error = np.abs(error) <= delta
    # For small errors: gradient = error (derivative of 0.5 * error^2)
    # For large errors: gradient = delta * sign(error) (derivative of delta * |error| - 0.5 * delta^2)
    gradient = np.where(is_small_error, error, delta * np.sign(error))
    return gradient

gradient = huber_gradient(y, yhat)

plt.figure(figsize=(10, 6))
plt.plot(theta, gradient)
plt.xlabel('theta')
plt.ylabel('Gradient of Huber Loss')
plt.title('Gradient of Huber Loss for y=0')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axhline(y=delta, color='r', linestyle='--', linewidth=0.5, label=f'delta = {delta}')
plt.axhline(y=-delta, color='r', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()




####################
#Question 3:
# Run the following code:
import tensorflow as tf
import numpy as np
tf.__version__
#'2.7.0'

ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2

inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = output)

#Compile the model with the huber_fn that you created in question 1 and then fit the model for 10 epochs, batch size 1.

#Answer:
delta = 5

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * tf.abs(error) - 0.5 * delta**2
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(optimizer='adam', loss=huber_fn)
model.fit(X, y, epochs=10, batch_size=1)


####################
#Question 4
#Run the following code:
y = np.arange(-3,3,0.01)
yhat = np.array([0]*len(y))

#Create two additional functions, mse_fn and mae_fn, and plot the MSE, MAE and Huber Loss in one plot.
# Title of the plot: Loss for y = 0 and different values of yhat
# X-label: yhat
# Y-label: Loss
#Answer:
delta = 5

def mse_fn(y_true, y_pred):
    return (y_true - y_pred)**2

def mae_fn(y_true, y_pred):
    return np.abs(y_true - y_pred)

def huber_fn_np(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * np.abs(error) - 0.5 * delta**2
    return np.where(is_small_error, squared_loss, linear_loss)

mse_loss = mse_fn(yhat, y)
mae_loss = mae_fn(yhat, y)
huber_loss = huber_fn_np(yhat, y)

plt.figure(figsize=(10, 6))
plt.plot(y, mse_loss, label='MSE')
plt.plot(y, mae_loss, label='MAE')
plt.plot(y, huber_loss, label='Huber Loss')
plt.xlabel('yhat')
plt.ylabel('Loss')
plt.title('Loss for y = 0 and different values of yhat')
plt.legend()
plt.grid(True)
plt.show()
