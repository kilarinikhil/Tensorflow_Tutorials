import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Define hyper parameters
numClasses = 10
numFeatures = 28*28
learningRate = 0.01
numSteps = 1000
batchSize = 256
displayStep = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the dataset to linear array
x_train = np.array(x_train.reshape([-1,numFeatures]),np.float32)
x_test = np.array(x_test.reshape([-1,numFeatures]),np.float32)  
x_train = x_train/255.
x_test = x_test/255.
print("Shape of x_train",x_train.shape)
print("Shape of y_train",y_train.shape)

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.repeat().shuffle(60000).batch(batchSize).prefetch(1)

# Define the variables
W = tf.Variable(tf.ones([numFeatures,numClasses]))
b = tf.Variable(tf.zeros([numClasses]))

# Define necessary functions
def logReg(x):
    return tf.nn.softmax(tf.matmul(x,W)+b)

def crossEntropy(y_pred,y_true):
    y_true = tf.one_hot(y_true,depth = numClasses,dtype = tf.float32)
    return tf.reduce_mean(-tf.math.log(tf.reduce_sum(tf.math.multiply(y_true,y_pred),1)))

def accuracy(y_pred,y_true):
    correctPredictions = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correctPredictions,tf.float32))

optimizer = tf.optimizers.SGD(learningRate)

def runOptimization(x,y):
    with tf.GradientTape() as g:
        pred = logReg(x)
        loss = crossEntropy(pred,y)

    # Compute gradients
    gradients = g.gradient(loss,[W,b])
    optimizer.apply_gradients(zip(gradients,[W,b]))

for step,(batchX,batchY) in enumerate(train_data.take(numSteps),1):
    runOptimization(batchX,batchY)

    if step % displayStep == 0:
        pred = logReg(batchX)
        loss = crossEntropy(pred,batchY)
        acc = accuracy(pred,batchY)
        print("Step: %i, loss: %f, accuracy: %f"%(step,loss,acc))

pred = logReg(x_test)
acc = accuracy(pred,y_test)
print("Your test data set accuracy is %f"%(acc))

nImages = 5
testImages = x_test[:nImages]

for i in range(nImages):
    plt.imshow(np.reshape(testImages[i],[28,28]),cmap = 'gray')
    plt.show()
    print("Model prediction: %i" %np.argmax(pred.numpy()[i]))
