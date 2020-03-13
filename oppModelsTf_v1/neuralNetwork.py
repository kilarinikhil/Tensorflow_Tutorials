import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = (x_train/255.).reshape(60000,784)
x_test = (x_test/255.).reshape(10000,784)

class NeuralNetwork:
	def __init__(self,x_train,y_train,x_test,y_test,learning_rate,batch_size,num_steps):
		self.nh1 = 500
		self.nh2 = 300
		self.out_size = 10
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_steps = num_steps
		
	def neu_net(self,x_dict):
		self.layer1 = tf.layers.dense(x_dict['images'],self.nh1)
		self.layer2 = tf.layers.dense(self.layer1,self.nh2)
		self.out_layer = tf.nn.softmax(tf.layers.dense(self.layer2,self.out_size))
		return self.out_layer
		
	def model_fn(self,features,labels,mode):
		self.logits = self.neu_net(features)
		self.classes = tf.math.argmax(self.logits,1)
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode = mode,predictions = self.classes)
		self.labelstologits = tf.cast(tf.one_hot(labels,10,axis = 1),dtype = tf.float64)
		self.loss_op = tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.log(self.logits),self.labelstologits),1))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss_op,global_step = tf.train.get_global_step())
		
		#Define accuracy model
		self.acc_op = tf.metrics.accuracy(labels = labels,predictions = self.classes)
		
		self.estim_spec = tf.estimator.EstimatorSpec(mode = mode,predictions = self.classes,loss = self.loss_op,train_op = self.train_op,eval_metric_ops = {'accuracy':self.acc_op}) 
		return self.estim_spec
	
	def train(self):
		self.model = tf.estimator.Estimator(self.model_fn)
		self.input_fn = tf.estimator.inputs.numpy_input_fn(x = {'images':self.x_train},y = self.y_train,batch_size = self.batch_size,shuffle = True)
		self.model.train(self.input_fn,steps = self.num_steps)
		
	def predict(self):#Returns predictions and confusion matrix for test dataset
		self.input_fn = tf.estimator.inputs.numpy_input_fn(x = {'images':self.x_test},y = self.y_test,shuffle = False,batch_size = self.batch_size)
		self.preds = list(self.model.predict(self.input_fn))
		self.conf_mat = tf.math.confusion_matrix(self.y_test,self.preds)
		print(tf.Session().run(self.conf_mat))
		return self.preds,self.conf_mat
		
#Testing the neural_network		 
nn1 = NeuralNetwork(x_train,y_train,x_test,y_test,0.01,128,1000)
nn1.train()
nn1.predict()

print("If you see a confusion matrix remove the last four lines of code and import the class directly")
