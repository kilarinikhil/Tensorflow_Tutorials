import tensorflow as tf
import numpy as np

#Borrowing dataset from tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

#Data preprocessing
#Normalizing the training data
x_train = (x_train/255.).reshape(60000,784)
y_train = tf.Session().run(tf.one_hot(y_train,10))
#Normalizing the test data
x_test = (x_test/255.).reshape(10000,784)
y_test = tf.Session().run(tf.one_hot(y_test,10))

X = tf.placeholder('float64', [None,784])
Y = tf.placeholder('float64', [None,10])

#Defining weights
ker = tf.Variable(tf.zeros([784,10],dtype = 'float64'), name = 'kernel1')
b   = tf.Variable(tf.zeros([10],dtype = 'float64'), name = 'bias1')

#Defining Hyperparameters
learning_rate = 0.01
epochs = 30
display_step = 5#Modify to get cost after certain epochs as per your requirements

#Define model
pred = tf.nn.softmax(tf.add(tf.matmul(X,ker),b))


#Define Cost
cost = tf.reduce_mean(-tf.reduce_sum(tf.multiply(Y,tf.log(pred)),1))

#Define Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	#Training
	for epoch in range(epochs):
		sess.run(optimizer,feed_dict = {X: x_train,Y: y_train})
		
		if (epoch+1) % display_step == 0:
			print("Epoch : ",epoch+1, "Cost = ", sess.run(cost, feed_dict = {X: x_train,Y: y_train}))
	
	print("Optimization Finished!")
	
	#Testing
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float64'))
	print("Accuracy = ",accuracy.eval({X:x_test, Y:y_test}))
