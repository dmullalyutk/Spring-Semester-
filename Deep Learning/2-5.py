import tensorflow as tf 



import matplotlib.pyplot as plt
import numpy as np

ct = np.ones(20)

X1= np.random.normal(size = 20)
X2= np.random.normal(size = 20)

X = np.array(np.column_stack((X1,X2)))
y = ct *2.222 +X1 *5.4657 + X2*10.115 - 3 * X1**2

inputs = tf.keras.layers.Input(shape = (X.shape[1],), name = 'inputs')
hidden1 = tf.keras.layers.Dense(units=2,activation='sigmoid',name = 'hidden1')(inputs)

hidden2 = tf.keras.layers.Dense(units=1,activation='linear',name = 'hidden2')(hidden1)

outputs = tf.keras.layers.Dense(units=1,activation='linear',name = 'outputs')(hidden2)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss = "mse", optimizer = tf.keras.optimizers.SGD(learning_rate=0.001))

model.fit(x=X,y=y, batch_size =1 , epochs = 10)

yhat = model.predict(x = X[0:1])

model.evaluate(X,y)

model.summary()


history = model.fit(x=X,y=y, batch_size =1 , epochs = 10)
import pandas as pd 
pd.DataFrame(history.history["loss"]).plot()
plt.show()

model.fit(x=X,y=y, batch_size =1 , epochs = 10, validation_data = (X,y))

model.fit(x=X,y=y, batch_size =1 , epochs = 10, validation_split=0.2)


