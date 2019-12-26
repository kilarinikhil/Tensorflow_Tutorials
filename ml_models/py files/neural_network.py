import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

input_size = 784
num_steps = 10
nh1 = 500
nh2 = 300
out_size = 10
num_steps = 100#number of epochs

#Define hyper parameters
learning_rate = 0.01
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
batch_size = 128
x_train = (x_train/255.).reshape(60000,784)
x_test = x_test/255.
def neuralNetwork(x_dict):
	#Define input placeholder
	X = x_dict['images']
	
	layer1 = tf.layers.dense(X,nh1)
	layer2 = tf.layers.dense(layer1,nh2)
	out_layer = tf.nn.softmax(tf.layers.dense(layer2,out_size))
	
	return out_layer
	
def model_fn(features,labels,mode):
	
	#Call neuralNetwork()
	logits = neuralNetwork(features)
	classes = tf.math.argmax(logits,1)
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode,predictions = classes)
		
	#Define loss_op and train_op for training
	labelstoLogits = tf.cast(tf.one_hot(labels,10,axis = 1),dtype = tf.float64)
	loss_op = tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.log(logits),labelstoLogits),1))
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
	
	#Define accuracy of the model
	acc_op = tf.metrics.accuracy(labels = labels, predictions = classes)
	
	#Define estimator spec
	
	estim_spec = tf.estimator.EstimatorSpec(mode = mode,predictions = classes, loss = loss_op, train_op = train_op,eval_metric_ops ={'accuracy':acc_op} )
	return estim_spec
	
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(x = {'images' : x_train },y = y_train,batch_size = batch_size,num_epochs = None,shuffle = True)
model.train(input_fn,steps = num_steps)
	
	
