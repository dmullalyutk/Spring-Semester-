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


