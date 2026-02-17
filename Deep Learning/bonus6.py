import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5,5,0.001)

#Question 1
#Manually implement and plot the sigmoid activation function.

#Answer:
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid)
plt.title('Sigmoid')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.show()

#Question 2:
#Manually implement and plot the Tanh activation function.

#Answer:
tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
plt.plot(z, tanh)
plt.title('Tanh')
plt.xlabel('z')
plt.ylabel('tanh(z)')
plt.show()

#Question 3:
#Manually implement and plot the ReLu activation function.

#Answer:
relu = np.maximum(0, z)
plt.plot(z, relu)
plt.title('ReLU')
plt.xlabel('z')
plt.ylabel('ReLU(z)')
plt.show()

#Question 4:
#Manually implement and plot the LReLu (alpha = 0.01) activation function.

#Answer:
alpha = 0.01
lrelu = np.where(z > 0, z, alpha * z)
plt.plot(z, lrelu)
plt.title('Leaky ReLU (alpha = 0.01)')
plt.xlabel('z')
plt.ylabel('LReLU(z)')
plt.show()

#Question 5:
#Manually implement and plot the ELU (alpha = 0.01) activation function.

#Answer:
alpha = 0.01
elu = np.where(z > 0, z, alpha * (np.exp(z) - 1))
plt.plot(z, elu)
plt.title('ELU (alpha = 0.01)')
plt.xlabel('z')
plt.ylabel('ELU(z)')
plt.show()

#Question 6:
#Manually implement and plot the softplus activation function. Also create a plot for the derivative of the softplus.

#Answer:
softplus = np.log(1 + np.exp(z))
softplus_derivative = 1 / (1 + np.exp(-z))  # derivative of softplus is the sigmoid

plt.subplot(1, 2, 1)
plt.plot(z, softplus)
plt.title('Softplus')
plt.xlabel('z')
plt.ylabel('softplus(z)')

plt.subplot(1, 2, 2)
plt.plot(z, softplus_derivative)
plt.title('Derivative of Softplus')
plt.xlabel('z')
plt.ylabel("softplus'(z)")

plt.tight_layout()
plt.show()
