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

nNodes_hl0 = 500
nNodes_hl1 = 500
nNodes_hl2 = 500

nClasses = 10
batchSize = 100

# height x width
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
	# input data*weights + biases
	hiddenLayer0 = {'weights' : tf.Variable(tf.random_normal([784, nNodes_hl0])),
	                'biases' : tf.Variable(tf.random_normal([nNodes_hl0]))}
	
	hiddenLayer1 = {'weights' : tf.Variable(tf.random_normal([nNodes_hl0, nNodes_hl1])),
	                'biases' : tf.Variable(tf.random_normal([nNodes_hl1]))}
	
	hiddenLayer2 = {'weights' : tf.Variable(tf.random_normal([nNodes_hl1, nNodes_hl2])),
	                'biases' : tf.Variable(tf.random_normal([nNodes_hl2]))}
	
	outputLayer  = {'weights' : tf.Variable(tf.random_normal([nNodes_hl2, nClasses])),
	                'biases' : tf.Variable(tf.random_normal([nClasses]))}
	
	l0 = tf.add(
	            tf.matmul(data, hiddenLayer0['weights']), hiddenLayer0['biases'])
	l0 = tf.nn.relu(l0)
	
	l1 = tf.add(tf.matmul(l0, hiddenLayer1['weights']), hiddenLayer1['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']),hiddenLayer2['biases'])
	l2 = tf.nn.relu(l2)
	
	lOut = tf.matmul(l2, outputLayer['weights']) + outputLayer['biases']
	
	return lOut
	
	
def train_neural_network(x):
	prediction = neural_network_model(x)
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
