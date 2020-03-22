import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Define Hyperparameters
numFeatures = 28*28
batchSize = 256
nHidden1 = 128
nHidden2 = 64
learningRate = 0.01
numSteps = 20000
displayStep = 1000

(xTrain,yTrain),(xTest,yTest) = mnist.load_data()
xTrain, xTest = xTrain.reshape([-1,numFeatures]).astype(np.float32), xTest.reshape([-1,numFeatures]).astype(np.float32)
xTrain, xTest = xTrain/255., xTest/255.

trainData = tf.data.Dataset.from_tensor_slices((xTrain,yTrain))
trainData = trainData.repeat().shuffle(60000).batch(batchSize)

testData = tf.data.Dataset.from_tensor_slices((xTest,yTest))
testData = testData.repeat().shuffle(10000).batch(batchSize).prefetch(1)

randomNormal = tf.initializers.RandomNormal()

weights = {
        'encoderH1' : tf.Variable(randomNormal([numFeatures,nHidden1])),
        'encoderH2' : tf.Variable(randomNormal([nHidden1,nHidden2])),
        'decoderH1' : tf.Variable(randomNormal([nHidden2,nHidden1])),
        'decoderH2' : tf.Variable(randomNormal([nHidden1,numFeatures])),
        }
biases = {
        'encoderH1' : tf.Variable(randomNormal([nHidden1])),
        'encoderH2' : tf.Variable(randomNormal([nHidden2])),
        'decoderH1' : tf.Variable(randomNormal([nHidden1])),
        'decoderH2' : tf.Variable(randomNormal([numFeatures])),
        }

def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoderH1']),biases['encoderH1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoderH2']),biases['encoderH2']))
    return layer2

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoderH1']),biases['decoderH1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decoderH2']),biases['decoderH2']))
    return layer2

def meanSquare(reconstructed,original):
    return tf.reduce_mean(tf.pow(original-reconstructed,2))

optimizer = tf.optimizers.Adam(learningRate)

def runOptimization(x):
    with tf.GradientTape() as g:
        reconstructedImage = decoder(encoder(x))
        loss = meanSquare(reconstructedImage,x)
    trainableVars = list(weights.values())+list(biases.values())
    gradients = g.gradient(loss,trainableVars)
    optimizer.apply_gradients(zip(gradients,trainableVars))

    return loss
    

for step,(batchX,_) in enumerate(trainData.take(numSteps),1):
    loss = runOptimization(batchX)

    if step % displayStep == 0:
        print("Step: %i, loss: %f"%(step,loss))


# Encode and decode images from test set and visualize their reconstruction.
nImages = 4
canvas_orig = np.empty((28 * nImages, 28 * nImages))
canvas_recon = np.empty((28 * nImages, 28 * nImages))
for i, (batch_x, _) in enumerate(testData.take(nImages)):
    # Encode and decode the digit image.
    reconstructed_images = decoder(encoder(batch_x))
    # Display original images.
    for j in range(nImages):
        # Draw the generated digits.
        img = batch_x[j].numpy().reshape([28, 28])
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
    # Display reconstructed images.
    for j in range(nImages):
        # Draw the generated digits.
        reconstr_img = reconstructed_images[j].numpy().reshape([28, 28])
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = reconstr_img

print("Original Images")
plt.figure(figsize=(nImages, nImages))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(nImages, nImages))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()
