import numpy as np 
ct = np.ones(20)
x = np.random.normal(size = 20)
ct_x = np.column_stack((ct, x))
y = ct * 2.22 + x * 5.4675
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()

from sklearn import linear_model 

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(ct_x,y).coef_

eps = 1e-5
W = np.array(np.random.normal(size =2 )).reshape(2,1)

def error ( weight,x,y):
    y_hat = np.matmul(x,weight).flatten()
    error = (y-y_hat)**2
    return error

error(W,ct_x[0],y[0])

eta = .01

for j in range(1000):
    for i in range(len(ct_x)):
        impact1 = (error(np.array([W[0]+eps,W[1]]),ct_x[i],y[i])-error(np.array([W[0]-eps,W[1]]),ct_x[i],y[i]))/(2*eps)
        impact2 = (error(np.array([W[0],W[1]+eps]),ct_x[i],y[i])-error(np.array([W[0],W[1]-eps]),ct_x[i],y[i]))/(2*eps)

        W[0] = W[0] -  eta*impact1
        W[1] = W[1] - eta* impact2

import matplotlib.pyplot as plt 
plt.scatter(x,y)

Xplot = np.arange(min(x),max(x),0.1)
yplot = []

for x in Xplot:
    yplot.append(W[0] + W[1]*x)

plt.scatter(x,y)
plt.plot(Xplot,yplot)
plt.show()


#nueral net from scatch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 100 
ct = np.ones(n)

X = np.random.normal(size = n)

X= X / 3.0 

ct_X = np.array(np.column_stack((ct,X)))

ct_X

y = ct * 1.2222 + X * 2.4675 -8 *X**2 
plt.scatter(X,y)
plt.show()


eps = 1e-5

W_i_h= np.array(np.random.normal(size = 4 )).reshape(2,2)

W_h_o= np.array(np.random.normal(size = 3 )).reshape(3,1)



def error (W_i_h,W_h_o,X,y):
    #compute weighted sum layer 1
    z_1and2 = np.matmul(X,W_i_h)
    #activate
    h_1and2 = 1/ (1 + np.exp(-z_1and2))
    #compute weighted sum output layer 
    h_out = np.column_stack((np.array([1.0]),h_1and2.reshape(1,2)))
    o =np.matmul(h_out, W_h_o)
    error = (y-o)**2
    return error.item() 

error (W_i_h,W_h_o,ct_X[0],y[0])

eta = 0.01

for j in range(1000):
    for i in range(len(ct_X)):
        # W_i_h gradients (2x2 = 4 weights)
        impact_ih_00 = (error(np.array([[W_i_h[0,0]+eps,W_i_h[0,1]],[W_i_h[1,0],W_i_h[1,1]]]),W_h_o,ct_X[i],y[i])-error(np.array([[W_i_h[0,0]-eps,W_i_h[0,1]],[W_i_h[1,0],W_i_h[1,1]]]),W_h_o,ct_X[i],y[i]))/(2*eps)
        impact_ih_01 = (error(np.array([[W_i_h[0,0],W_i_h[0,1]+eps],[W_i_h[1,0],W_i_h[1,1]]]),W_h_o,ct_X[i],y[i])-error(np.array([[W_i_h[0,0],W_i_h[0,1]-eps],[W_i_h[1,0],W_i_h[1,1]]]),W_h_o,ct_X[i],y[i]))/(2*eps)
        impact_ih_10 = (error(np.array([[W_i_h[0,0],W_i_h[0,1]],[W_i_h[1,0]+eps,W_i_h[1,1]]]),W_h_o,ct_X[i],y[i])-error(np.array([[W_i_h[0,0],W_i_h[0,1]],[W_i_h[1,0]-eps,W_i_h[1,1]]]),W_h_o,ct_X[i],y[i]))/(2*eps)
        impact_ih_11 = (error(np.array([[W_i_h[0,0],W_i_h[0,1]],[W_i_h[1,0],W_i_h[1,1]+eps]]),W_h_o,ct_X[i],y[i])-error(np.array([[W_i_h[0,0],W_i_h[0,1]],[W_i_h[1,0],W_i_h[1,1]-eps]]),W_h_o,ct_X[i],y[i]))/(2*eps)

        # W_h_o gradients (3x1 = 3 weights)
        impact_ho_0 = (error(W_i_h,np.array([[W_h_o[0,0]+eps],[W_h_o[1,0]],[W_h_o[2,0]]]),ct_X[i],y[i])-error(W_i_h,np.array([[W_h_o[0,0]-eps],[W_h_o[1,0]],[W_h_o[2,0]]]),ct_X[i],y[i]))/(2*eps)
        impact_ho_1 = (error(W_i_h,np.array([[W_h_o[0,0]],[W_h_o[1,0]+eps],[W_h_o[2,0]]]),ct_X[i],y[i])-error(W_i_h,np.array([[W_h_o[0,0]],[W_h_o[1,0]-eps],[W_h_o[2,0]]]),ct_X[i],y[i]))/(2*eps)
        impact_ho_2 = (error(W_i_h,np.array([[W_h_o[0,0]],[W_h_o[1,0]],[W_h_o[2,0]+eps]]),ct_X[i],y[i])-error(W_i_h,np.array([[W_h_o[0,0]],[W_h_o[1,0]],[W_h_o[2,0]-eps]]),ct_X[i],y[i]))/(2*eps)

        # Update W_i_h
        W_i_h[0,0] = W_i_h[0,0] - eta*impact_ih_00
        W_i_h[0,1] = W_i_h[0,1] - eta*impact_ih_01
        W_i_h[1,0] = W_i_h[1,0] - eta*impact_ih_10
        W_i_h[1,1] = W_i_h[1,1] - eta*impact_ih_11

        # Update W_h_o
        W_h_o[0,0] = W_h_o[0,0] - eta*impact_ho_0
        W_h_o[1,0] = W_h_o[1,0] - eta*impact_ho_1
        W_h_o[2,0] = W_h_o[2,0] - eta*impact_ho_2

# Plot results
Xplot = np.linspace(min(X), max(X), 100)
yplot = []

for x_val in Xplot:
    x_input = np.array([1.0, x_val])
    z_1and2 = np.matmul(x_input, W_i_h)
    h_1and2 = 1 / (1 + np.exp(-z_1and2))
    h_out = np.array([1.0, h_1and2[0], h_1and2[1]])
    o = np.matmul(h_out, W_h_o)
    yplot.append(o.item())

plt.scatter(X, y, label='Data')
plt.plot(Xplot, yplot, 'r-', label='NN Prediction')
plt.legend()
plt.show()
