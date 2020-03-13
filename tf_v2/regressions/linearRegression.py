import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([1.2,1.5,2.6,3.4,5.4,6.1,7.8,9.1,10.3,11.1,12.3])
Y = np.array([5.2,6.4,7.1,8.2,9.1,10.3,11.2,13.7,14.7,15.2,16.7])
n_samples = X.shape[0]

# Declare variables
W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# Define hyperparameters
learningRate = 0.01
numEpochs = 1000
displayStep = 50

def linearReg(x):
	# Define the model
	y = W*x + b
	return y
	
def meanSquareErr(yPred,yTrue):
	mse = tf.reduce_sum(tf.pow((yPred-yTrue),2))/(2*n_samples)
	return mse

# define optimizer
optimizer = tf.optimizers.SGD(learningRate)

def runOptimization():
	with tf.GradientTape() as g:
		pred = linearReg(X)
		loss = meanSquareErr(pred,Y)
	
	# Compute gradients
	gradients = g.gradient(loss,[W,b])
	
	# Update W and b following gradients
	optimizer.apply_gradients(zip(gradients,[W,b]))
	
for step in range(1,numEpochs+1):
	runOptimization()
	if step % displayStep == 0:
		pred = linearReg(X)
		loss = meanSquareErr(pred,Y)
		print("Step: %i, loss: %f, W: %f, b: %f" %(step, loss, W.numpy(), b.numpy()))

plt.plot(X,Y,'ro',label = 'Original data')
plt.plot(X,np.array(W*X+b),label = 'Fit data')
plt.legend()
plt.show()
