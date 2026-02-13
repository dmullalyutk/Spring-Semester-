#python3 -m pip install --upgrade tensorflow
import tensorflow as tf
tf.__version__
import numpy as np
################################################################################################
#Question 1: Consider the following unactivated weighted sums for each of 6 output classes
unactivated_weighted_sum = tf.constant([10.0,10.0,5.0,6.0,8.0,13.0])
unactivated_weighted_sum
#The output of the softmax function is the following:
softmax_output = np.round(tf.nn.softmax(unactivated_weighted_sum),3)
softmax_output

#Manually code the softmax function, and test whether it returns the same output

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)  # numeric stability
    exps = np.exp(x)
    return exps / np.sum(exps)

manual_softmax = np.round(softmax(unactivated_weighted_sum), 3)
manual_softmax



################################################################################################
#Question 2: compute the sum of the values in unactivated_weighted_sum and the sum of the values in softmax_output. Is the result expected?
sum_unactivated = tf.reduce_sum(unactivated_weighted_sum).numpy()
sum_softmax = np.sum(softmax_output)
sum_unactivated, sum_softmax
# Answer: The unactivated sum can be any real value, while softmax outputs form a probability distribution that sums to 1 (up to rounding), which is expected.



################################################################################################
#Question 3: #Consider the following change in the unactivated_weighted_sum array (the first element is changed)
unactivated_weighted_sum2 = tf.constant([20.0,10.0,5.0,6.0,8.0,13.0])

softmax_output2 = np.round(tf.nn.softmax(unactivated_weighted_sum2),3)

#Plot the unactivated_weighted_sum and unactivated_weighted_sum2
#Plot the softmax_output and softmax_output2
#What do you observe and provide reasoning?

import matplotlib.pyplot as plt

labels = np.arange(len(unactivated_weighted_sum))

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.bar(labels-0.15, unactivated_weighted_sum.numpy(), width=0.3, label='sum1')
plt.bar(labels+0.15, unactivated_weighted_sum2.numpy(), width=0.3, label='sum2')
plt.title('Unactivated Weighted Sums')
plt.legend()

plt.subplot(1,2,2)
plt.bar(labels-0.15, softmax_output, width=0.3, label='softmax1')
plt.bar(labels+0.15, softmax_output2, width=0.3, label='softmax2')
plt.title('Softmax Outputs')
plt.legend()
plt.tight_layout()
plt.show()

#Answer: Increasing one logit (first element) sharply increases its softmax probability while decreasing the others. Softmax is sensitive to relative differences and exponentiates logits, so a larger logit dominates the distribution.

    
