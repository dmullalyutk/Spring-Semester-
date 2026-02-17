import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5,5,0.001)

#Question 1
#Manually implement and plot the sigmoid activation function.
sigmoid = 1 / (1 + np.exp(-z))
plt.figure()
plt.plot(z, sigmoid)
plt.title('Sigmoid Activation Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.savefig('sigmoid.png')
plt.show()

#Question 2:
#Manually implement and plot the Tanh activation function.
tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
plt.figure()
plt.plot(z, tanh)
plt.title('Tanh Activation Function')
plt.xlabel('z')
plt.ylabel('tanh(z)')
plt.grid(True)
plt.savefig('tanh.png')
plt.show()

#Question 3:
#Manually implement and plot the ReLu activation function.
relu = np.maximum(0, z)
plt.figure()
plt.plot(z, relu)
plt.title('ReLU Activation Function')
plt.xlabel('z')
plt.ylabel('ReLU(z)')
plt.grid(True)
plt.savefig('relu.png')
plt.show()

#Question 4:
#Manually implement and plot the LReLu (alpha = 0.01) activation function.
alpha_lrelu = 0.01
lrelu = np.where(z > 0, z, alpha_lrelu * z)
plt.figure()
plt.plot(z, lrelu)
plt.title('Leaky ReLU Activation Function (alpha = 0.01)')
plt.xlabel('z')
plt.ylabel('LReLU(z)')
plt.grid(True)
plt.savefig('lrelu.png')
plt.show()

#Question 5:
#Manually implement and plot the ELU (alpha = 0.01) activation function.
alpha_elu = 0.01
elu = np.where(z > 0, z, alpha_elu * (np.exp(z) - 1))
plt.figure()
plt.plot(z, elu)
plt.title('ELU Activation Function (alpha = 0.01)')
plt.xlabel('z')
plt.ylabel('ELU(z)')
plt.grid(True)
plt.savefig('elu.png')
plt.show()

#Question 6:
#Manually implement and plot the softplus activation function. Also create a plot for the derivative of the softplus.
softplus = np.log(1 + np.exp(z))
softplus_derivative = 1 / (1 + np.exp(-z))  # derivative of softplus is the sigmoid

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(z, softplus)
ax1.set_title('Softplus Activation Function')
ax1.set_xlabel('z')
ax1.set_ylabel('Softplus(z)')
ax1.grid(True)

ax2.plot(z, softplus_derivative)
ax2.set_title('Derivative of Softplus (Sigmoid)')
ax2.set_xlabel('z')
ax2.set_ylabel("Softplus'(z)")
ax2.grid(True)

plt.tight_layout()
plt.savefig('softplus.png')
plt.show()
