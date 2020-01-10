import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

#normalize dataset and reshape
x_train = (x_train/255.).reshape(60000,784)

#hyper parameters
numEpochs = 140
learningRate = 0.0002
batchSize = 120
displayStep = 10
batches = int(60000/batchSize)

imageSize = 784
genHidSize = 256
discHidSize = 256
noiseSize = 100

#def glorit initialization
def gloritInit(shape):
	return tf.random.normal(shape,stddev = 1. / tf.sqrt(shape[0]/2))

#Intiating weights	
weights = {
	'genHid1' : tf.Variable(gloritInit([noiseSize,genHidSize])),
	'genOut'  : tf.Variable(gloritInit([genHidSize,imageSize])),
	'discHid1': tf.Variable(gloritInit([imageSize,discHidSize])),
	'discOut' : tf.Variable(gloritInit([discHidSize,1]))
}

biases = {
	'genHid1' : tf.Variable(tf.zeros([genHidSize])),
	'genOut'  : tf.Variable(tf.zeros([imageSize])),
	'discHid1': tf.Variable(tf.zeros([discHidSize])),
	'discOut' : tf.Variable(tf.zeros([1]))
}

#Define Generator
def generator(x):
	#Define generator model
	hiddenlayer = tf.nn.relu(tf.add(tf.matmul(x,weights['genHid1']),biases['genHid1']))
	
	outlayer = tf.nn.sigmoid(tf.add(tf.matmul(hiddenlayer,weights['genOut']),biases['genOut']))
	
	return outlayer

#Define discriminator
def discriminator(x):
	#Define discriminator model
	hiddenlayer = tf.nn.relu(tf.add(tf.matmul(x,weights['discHid1']),biases['discHid1']))
	
	outlayer = tf.nn.sigmoid(tf.add(tf.matmul(hiddenlayer,weights['discOut']),biases['discOut']))
	
	return outlayer
	
#Define placeholders for generator and discriminator
gen_in = tf.placeholder(tf.float32, shape = [None, noiseSize])
disc_in = tf.placeholder(tf.float32, shape = [None, imageSize])

#Input from dataset
discReal = discriminator(disc_in)

#Input from generated sample from noise
generatedSample = generator(gen_in)
discFake = discriminator(generatedSample)

#Define loss operators
gen_loss = -tf.reduce_mean(tf.log(discFake))
disc_loss = - tf.reduce_mean(tf.log(discReal) + tf.log(1. - discFake))

#Define optimizers
optimizerGen = tf.train.AdamOptimizer(learningRate)
optimizerDisc = tf.train.AdamOptimizer(learningRate)

#Define var to be optimized by each optimizer as by default all variables are optimized for a define optimizer
genVars = [weights['genHid1'],weights['genOut'],biases['genHid1'],biases['genOut']]
disVars = [weights['discHid1'],weights['discOut'],biases['discHid1'],biases['discOut']]

train_gen = optimizerGen.minimize(gen_loss,var_list = genVars)
train_disc = optimizerDisc.minimize(disc_loss,var_list = disVars)

#Initiate variables 
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for i in range(numEpochs):
    for j in range(batches):
      noise = np.random.uniform(-1., 1.,size = [batchSize,noiseSize])
      batchX = x_train[j*batchSize:(j+1)*batchSize,:]
      #Start training
      feed_dict = {gen_in:noise, disc_in:batchX}

      _, _, gl, dl = sess.run([train_gen,train_disc,gen_loss,disc_loss],feed_dict = feed_dict)

    if i % displayStep == 0 or i == 1:
      print("Epoch : %i Generator loss : %f Discriminator loss %f"%(i,gl,dl))

  # Testing
  # Generate images from noise, using the generator network.
  n = 6
  canvas = np.empty((28 * n, 28 * n))
  for i in range(n):
      # Noise input.
      z = np.random.uniform(-1., 1., size=[n, noiseSize])
      # Generate image from noise.
      g = sess.run(generatedSample, feed_dict={gen_in: z})
      # Reverse colours for better display
      g = -1 * (g - 1)
      for j in range(n):
          # Draw the generated digits
          canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

  plt.figure(figsize=(n, n))
  plt.imshow(canvas, origin="upper", cmap="gray")
  plt.show()
