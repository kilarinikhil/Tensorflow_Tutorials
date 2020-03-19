import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Define hyperparameters
learningRate = 0.001
trainingSteps = 200
batchSize = 128
displayStep = 10
numClasses = 10
numFeatures = 28*28
conv1Filters = 32
conv2Filters = 64
fc1Units = 1024

(xTrain,yTrain),(xTest,yTest) = mnist.load_data()
xTrain, xtest = np.array(xTrain,np.float32),np.array(xTest,np.float32)
xTrain,xTest = xTrain/255.,xTest/255.

trainData = tf.data.Dataset.from_tensor_slices((xTrain,yTrain))
trainData = trainData.repeat().shuffle(60000).batch(batchSize).prefetch(1)

class convNet(Model):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1 = layers.Conv2D(conv1Filters, kernel_size = 5, activation = tf.nn.relu)
        self.maxPool1 = layers.MaxPool2D(2,strides = 2)
        self.conv2 = layers.Conv2D(conv2Filters, kernel_size = 3, activation = tf.nn.relu)
        self.maxPool2 = layers.MaxPool2D(2,strides = 2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(fc1Units)
        self.dropout = layers.Dropout(rate = 0.5)
        self.out = layers.Dense(numClasses)
        
    def call(self,x,isTraining = False):
        x = tf.reshape(x,[-1,28,28,1])
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training = isTraining)
        x = self.out(x)
        if not isTraining:
            x = tf.nn.softmax(x)
        return x

conv_net = convNet()

def crossEntropy(x,y):
    y = tf.cast(y,tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    return tf.reduce_mean(loss)

def accuracy(yPred,yTrue):
   correctPredictions = tf.equal(tf.argmax(yPred,1),tf.cast(yTrue,tf.int64))
   return tf.reduce_mean(tf.cast(correctPredictions,tf.float32))

optimizer = tf.optimizers.Adam(learningRate)

def runOptimization(x,y):
    with tf.GradientTape() as g:
        pred = conv_net(x,isTraining = True)
        loss = crossEntropy(pred,y)
    trainableVars = conv_net.trainable_variables
    gradients = g.gradient(loss,trainableVars)
    optimizer.apply_gradients(zip(gradients,trainableVars))

for step,(batchX,batchY) in enumerate(trainData.take(trainingSteps),1):
    runOptimization(batchX,batchY)

    if step % displayStep == 0:
        pred = conv_net(batchX)
        loss = crossEntropy(pred,batchY)
        acc = accuracy(pred,batchY)
        print("Step: %i, loss: %f, acc: %f"%(step,loss,acc))

pred = conv_net(xTest)
acc = accuracy(pred,yTest)
print("Test set accuracy: %f"%(acc))

nImages = 5
for i in range(nImages):
    plt.imshow(np.reshape(xTest[i],[28,28]),cmap = 'gray')
    plt.show()
    print("Model prediction: %i"%(np.argmax(pred.numpy()[i])))
