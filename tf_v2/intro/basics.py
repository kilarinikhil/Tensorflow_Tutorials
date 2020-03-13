import tensorflow as tf

a = tf.constant(3.)
b = tf.constant(5.)
c = tf.constant(-7.)

add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.divide(a,b)

print(add.numpy())
print(sub.numpy())
print(mul.numpy())
print(div.numpy())

mean = tf.reduce_mean([a,b,c])
sum = tf.reduce_sum([a,b,c])

print(mean.numpy())
print(sum.numpy())

d = tf.constant([[1.,3.],[1.,2.]])
e = tf.constant([[2.,7.],[3.,5.]])

add = tf.add(d,e)
mul = tf.multiply(d,e)

print(add.numpy())
print(mul.numpy())
