import tensorflow as tf 

import matplotlib.pyplot as plt
import numpy as np 

ct = np.ones(20)
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows

X = np.array(np.column_stack((X1,X2)))

y1 = ct*2.22222 + X1*5.4675 + X2 * 10.115 - 3*X1**2
y2 = ct*2.22222 + X1*4.4675 + X2 * 10.45 - 6*X1**2



inputs = tf.keras.layers.Input(shape=(X.shape[1],), name = 'input layer')
hiden1 = tf.keras.layers.Dense(units=2,activation="sigmoid",name = "hidden1")(inputs)
hidden2 = tf.keras.layers.Dense(units=2,activation="sigmoid",name = "hidden2")(hiden1)
output1 = tf.keras.layers.Dense(units=1,activation="linear",name = "output1")(hidden2)
output2 = tf.keras.layers.Dense(units=1,activation="linear",name = "output2")(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = [output1, output2])
model.compile(loss='mse',optimizer= tf.keras.optimizers.SGD(learning_rate=0.01))

model.fit(X, [y1,y2], epochs=10, batch_size=1)

yhat1, yhat2 = model.predict(X[0:1])
model.evaluate(X, [y1,y2], batch_size=1)
model.summary()



