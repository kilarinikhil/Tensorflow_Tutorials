import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Model

# Preprocess data
(xTrain,yTrain),(xTest,yTest) = mnist.load_data()

# Define hyper parameters
nh1 = 256
nh2 = 128
numFeatures = 28*28
numClasses = 10
learningRate = 0.01
numSteps = 2000
batchSize = 256
displayStep = 100

xTrain = np.array(xTrain.reshape([-1,numFeatures]),dtype = np.float32)/255.
xTest = np.array(xTest.reshape([-1,numFeatures]),dtype = np.float32)/255.
trainData = tf.data.Dataset.from_tensor_slices((xTrain,yTrain))
trainData = trainData.repeat().shuffle(60000).batch(batchSize).prefetch(1)

# Create TF Model
class neuralNet(Model):
    def __init__(self):
        super(neuralNet,self).__init__()
        self.fc1 = layers.Dense(nh1, activation = tf.nn.relu)
        self.fc2 = layers.Dense(nh2, activation = tf.nn.relu)
        self.out = layers.Dense(numClasses)

    def call(self,x, isTraining = False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if not isTraining:
            x = tf.nn.softmax(x)

        return x

def crossEntropyLoss(x,y):
    y = tf.cast(y,tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    return tf.reduce_mean(loss)

def accuracy(yPred,yTrue):
    correctPredictions = tf.equal(tf.argmax(yPred,-1),tf.cast(yTrue,tf.int64))
    return tf.reduce_mean(tf.cast(correctPredictions,tf.float32))

optimizer = tf.optimizers.SGD(learningRate)
# Create a neuralNet object
neural_network = neuralNet()

def runOptimization(x,y):
    with tf.GradientTape() as g:
        pred = neural_network(x, isTraining = True)
        loss = crossEntropyLoss(pred,y)
    
    trainableVars = neural_network.trainable_variables
    gradients = g.gradient(loss,trainableVars)
    optimizer.apply_gradients(zip(gradients,trainableVars))

for step,(batchX,batchY) in enumerate(trainData.take(numSteps),1):
    runOptimization(batchX,batchY)
    if step % displayStep == 0:
        pred = neural_network(batchX,isTraining = False)
        loss = crossEntropyLoss(pred,batchY)
        acc = accuracy(pred,batchY)
        print("Step: %i, loss: %f, accuracy: %f"%(step, loss,acc))

pred = neural_network(xTest)
acc = accuracy(pred,yTest)
print("Test accuracy is %f"%(acc))

nImages = 5
testImages = xTest[:nImages]
for i in range(nImages):
    plt.imshow(np.reshape(testImages[i],[28,28]),cmap = 'gray')
    plt.show()
    print("Model prediction: %i"%np.argmax(pred[:5].numpy()[i]))
