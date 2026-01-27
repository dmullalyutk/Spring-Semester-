#Bonus 1

##############################################################
#Question 1: Variable Importance

#Run the following code, it will create some toy data and a linear model.

#Create some toy data. Set b=2.2222 and X=5.4675
import numpy as np
cte = np.ones(20) 
X1 = np.random.normal(size=20) #one variable, 20 rows
X2 = np.random.normal(size=20) #one variable, 20 rows
cte_X = np.array(np.column_stack((cte,X1,X2))) #combine cte and X
noise = 5*np.random.normal(size=20)
y = cte*2.2222 + X1*5.4675 + X2*10.1457 + noise

#Estimate a linear model
from sklearn import linear_model
# specify model
# we set fit_intercept to false because we manually added the 1 vector
reg = linear_model.LinearRegression(fit_intercept=False) 
#fit regression and extract coefficients
model = reg.fit(cte_X,y)
#make prediction
model.predict(cte_X)

#Next, compute the variable importance of X1 and X2 from step 2 onwards. Use the correlation between y and yhat as performance measure.



#Answer:

# Step 1: Get baseline performance (correlation between y and yhat)
yhat = model.predict(cte_X)
baseline_corr = np.corrcoef(y, yhat)[0, 1]
print(f"Baseline correlation: {baseline_corr:.4f}")

# Step 2: Variable importance for X1 (permutation importance)
# Shuffle X1 and measure performance drop
np.random.seed(42)
X1_shuffled = np.random.permutation(X1)
cte_X_permuted_X1 = np.column_stack((cte, X1_shuffled, X2))
yhat_permuted_X1 = model.predict(cte_X_permuted_X1)
corr_X1_permuted = np.corrcoef(y, yhat_permuted_X1)[0, 1]
importance_X1 = baseline_corr - corr_X1_permuted
print(f"Variable importance of X1: {importance_X1:.4f}")

# Step 3: Variable importance for X2 (permutation importance)
# Shuffle X2 and measure performance drop
X2_shuffled = np.random.permutation(X2)
cte_X_permuted_X2 = np.column_stack((cte, X1, X2_shuffled))
yhat_permuted_X2 = model.predict(cte_X_permuted_X2)
corr_X2_permuted = np.corrcoef(y, yhat_permuted_X2)[0, 1]
importance_X2 = baseline_corr - corr_X2_permuted
print(f"Variable importance of X2: {importance_X2:.4f}")




##############################################################
#Question 2: partial dependence plots

#Run the following code, it will create some toy data and a linear model.

#Create some toy data. Set b=2.2222 and X=5.4675
import numpy as np
cte = np.ones(20) 
X1 = np.random.normal(size=20) #one variable, 20 rows
X2 = np.random.normal(size=20) #one variable, 20 rows
cte_X = np.array(np.column_stack((cte,X1,X2))) #combine cte and X
noise = 5*np.random.normal(size=20)
y = cte*2.2222 + X1*5.4675 + X2*10.1457 + noise

#Estimate a linear model
from sklearn import linear_model
# specify model
# we set fit_intercept to false because we manually added the 1 vector
reg = linear_model.LinearRegression(fit_intercept=False) 
#fit regression and extract coefficients
model = reg.fit(cte_X,y)
#make prediction
model.predict(cte_X)

#Next, create a partial dependence plot for X1 and a partial dependence plot for X2

#Answer:
import matplotlib.pyplot as plt

# Compute PDP for X1
X1_range = np.linspace(X1.min(), X1.max(), 50)
pdp_X1 = []
for x1_val in X1_range:
    cte_X_temp = np.column_stack((cte, np.full(20, x1_val), X2))
    pdp_X1.append(np.mean(model.predict(cte_X_temp)))

# Compute PDP for X2
X2_range = np.linspace(X2.min(), X2.max(), 50)
pdp_X2 = []
for x2_val in X2_range:
    cte_X_temp = np.column_stack((cte, X1, np.full(20, x2_val)))
    pdp_X2.append(np.mean(model.predict(cte_X_temp)))

# Calculate common axis limits for comparison
x_min = min(X1_range.min(), X2_range.min())
x_max = max(X1_range.max(), X2_range.max())
y_min = min(min(pdp_X1), min(pdp_X2))
y_max = max(max(pdp_X1), max(pdp_X2))

# Plot with same axes
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(X1_range, pdp_X1)
plt.xlabel('X1')
plt.ylabel('Partial Dependence')
plt.title('Partial Dependence Plot for X1')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.subplot(1, 2, 2)
plt.plot(X2_range, pdp_X2)
plt.xlabel('X2')
plt.ylabel('Partial Dependence')
plt.title('Partial Dependence Plot for X2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()




##############################################################
#Question 3: Matrix multiply the following two matrices:
import numpy as np    
#X contains three records of data, for five features
X = np.array([[1,2,3,4,5],
              [4,5,6,7,8],
              [9,10,11,12,13]])
X.shape
#W contains weights for five features connected to two hidden units
W = np.array([[0.1,0.2],
              [0.3,0.4],
              [0.5,0.6],
              [0.7,0.8],
              [0.9,1]])
W.shape

#Answer:
# Matrix multiplication: X (3x5) @ W (5x2) = result (3x2)
result = np.matmul(X, W)
# Or equivalently: result = X @ W
print("Result of X @ W:")
print(result)
print(f"Result shape: {result.shape}")

##############################################################
########Question 4
#Consider the following loss curve.

import matplotlib.pyplot as plt
import numpy as np

def error_fn(x):
    return (x - 5)**2
    
y = []    
for x in range(10):
    y.append(error_fn(x))
    
plt.plot(y)
plt.xlabel('Weight')
plt.ylabel('Loss function')
plt.show()

#Next, consider the starting weight w, eta, and h.
#Implement the learning rule 20 times in a for loop and add a line to the plot above showing how the weight evolves.

w = 1 #initial weight
eta = 0.05
h = 0.1

#Answer:
# Store weight history for plotting
weight_history = [w]

# Implement gradient descent learning rule 20 times
for i in range(20):
    # Compute numerical gradient using finite differences
    # gradient ≈ (f(w+h) - f(w-h)) / (2*h)
    gradient = (error_fn(w + h) - error_fn(w - h)) / (2 * h)

    # Update weight using gradient descent rule
    w = w - eta * gradient

    # Store weight for plotting
    weight_history.append(w)

print(f"Final weight: {w:.4f} (optimal is 5)")

# Plot the loss curve and weight evolution
plt.figure(figsize=(10, 5))

# Plot loss function
x_vals = np.linspace(0, 10, 100)
y_vals = [error_fn(x) for x in x_vals]
plt.plot(x_vals, y_vals, 'b-', label='Loss function')

# Plot weight evolution as points on the loss curve
loss_at_weights = [error_fn(w) for w in weight_history]
plt.plot(weight_history, loss_at_weights, 'ro-', markersize=5, label='Weight evolution')

# Mark start and end points
plt.plot(weight_history[0], loss_at_weights[0], 'go', markersize=10, label=f'Start (w={weight_history[0]:.2f})')
plt.plot(weight_history[-1], loss_at_weights[-1], 'r*', markersize=15, label=f'End (w={weight_history[-1]:.2f})')

plt.xlabel('Weight')
plt.ylabel('Loss function')
plt.title('Gradient Descent: Weight Evolution on Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
