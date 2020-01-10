from __future__ import division, print_function, absolute_import
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = (x_train/255.).reshape(60000,784)
x_test = (x_test/255.).reshape(10000,784)

#hyperparameters
learningRate = 0.01
numEpochs = 5000
displayStep = 100
inputSize = 784

#encoder specifications
enchidden1Size = 256
enchidden2Size = 128

#decoder specifications
dechidden2Size = 256

X = tf.placeholder("float",[None,inputSize])

weights = {
	'encW1' : tf.Variable(tf.random.normal([inputSize,enchidden1Size])),
	'encW2' : tf.Variable(tf.random.normal([enchidden1Size,enchidden2Size])),
	'decW1' : tf.Variable(tf.random.normal([enchidden2Size,dechidden2Size])),
	'decW2' : tf.Variable(tf.random.normal([dechidden2Size,inputSize]))
}

biases = {
	'encb1' : tf.Variable(tf.random.normal([enchidden1Size])),
	'encb2' : tf.Variable(tf.random.normal([enchidden2Size])),
	'decb1' : tf.Variable(tf.random.normal([dechidden2Size])),
	'decb2' : tf.Variable(tf.random.normal([inputSize]))
}

def encoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encW1']),biases['encb1']))	
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encW2']),biases['encb2']))
	
	return layer2

def decoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decW1']),biases['decb1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decW2']),biases['decb2']))
	
	return layer2

#define operations
encoderop = encoder(X)
decoderop = decoder(encoderop)

ytrue = X
ypred = decoderop

#define loss and optimization operator
lossop = tf.reduce_mean(tf.pow((ytrue-ypred),2))
optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(lossop)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(1,numEpochs+1):
		_,l = sess.run([optimizer,lossop], feed_dict = {X:x_train})
		
		if(i%displayStep == 0 or i == 1):
			print('Epoch %i : Loss:%f'%(i,l))
	g = sess.run(decoderop,feed_dict = {X:x_test})

#Display first 4 images in the test set\
fig=plt.figure(figsize=(10,10))
for i in range(4):
	gsample = g[i].reshape(28,28)
	fig.add_subplot(2,2,i+1)
	plt.imshow(gsample,cmap = "gray")

plt.show()
