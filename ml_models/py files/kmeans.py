import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

#Borrowing dataset from tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

#Normalizing the data
x_train = x_train/255.
x_test = x_test/255.

y_train = tf.Session().run(tf.one_hot(y_train,10))
y_test = tf.Session().run(tf.one_hot(y_test,10))
