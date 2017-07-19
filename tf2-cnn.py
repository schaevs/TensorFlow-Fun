import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

'''
input > weight > hidden layer 1 (activation fn) > weights 
> hidden layer 2 (activation function) > weights > output layer 

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer ... SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch

'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

nClasses = 10
batchSize = 128
keepRate = 0.8
keepProb = tf.placeholder(tf.float32)

# height x width
x = tf.placeholder('float',[None,784])
y = tf.placeholder(tf.float64)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME' )
	
def maxpool2d(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], 
	                      strides = [1,2,2,1], padding = 'SAME')
	
def convolutional_neural_network(x):
	# input data*weights + biases
	weights = {'wConv0' : tf.Variable(tf.random_normal([5,5,1,32])),
	           'wConv1' : tf.Variable(tf.random_normal([5,5,32,64])),
	           'wFC' : tf.Variable(tf.random_normal([7*7*64,1024])),
	           'out' : tf.Variable(tf.random_normal([1024, nClasses]))
	           }
	
	biases = {'bConv0' : tf.Variable(tf.random_normal([32])),
	           'bConv1' : tf.Variable(tf.random_normal([64])),
	           'bFC' : tf.Variable(tf.random_normal([1024])),
	           'out' : tf.Variable(tf.random_normal([nClasses]))
	          }
	
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	conv0 = conv2d(x, weights['wConv0'])
	conv0 = maxpool2d(conv0)
	
	conv1 = conv2d(conv0, weights['wConv1'])
	conv1 = maxpool2d(conv1)
	
	fc = tf.reshape(conv1, [-1, 7*7*64])
	fc = tf.nn.relu( tf.matmul(fc, weights['wFC'])  + biases['bFC'])
	fc = tf.nn.dropout(fc, keepRate)
	
	lOut = tf.matmul(fc, weights['out']) + biases['out']
	
	return lOut
	
	
def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
	
	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	nEpochs = 10
	
	with tf.Session() as sesh:
		sesh.run(tf.global_variables_initializer())
		
		for epoch in range(nEpochs):
			epochLoss = 0
			for _ in range(int(mnist.train.num_examples/batchSize) ):
				ex, ey = mnist.train.next_batch(batchSize)
				_, c = sesh.run([optimizer, cost], feed_dict = {x: ex, y: ey})
				epochLoss += c
			print('Epoch', epoch, ' of ', nEpochs, 'loss: ', epochLoss)
			
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		
train_neural_network(x)
