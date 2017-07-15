import pandas as pd
import numpy as np
import bigfloat
#input layer size is 784 pixels and output layer size is 10 classes

#Reading a csv file
df = pd.read_csv("train.csv")
output = np.array(df['label'])
gr_descent = 0.0000001
lmbda = 0.00001
row,col = df.shape
train_size = row
pixel_cnt = col - 1
# print train_size, pixel_cnt
train_data = df.as_matrix()
# print train_data.shape
train_data = np.delete(train_data,0,1)
# print train_data.shape
# print output

#Calculating loss on the dataset
#hiddenlayers is an array
#input layer is the dimension
#I am making two hidden layers in neural networks


def test_data(model):
	df2 = pd.read_csv("test.csv")
	input_layer = df2.as_matrix()
	W1 = model['W1']
	W2 = model['W2']
	b1 = model['b1']
	b2 = model['b2']
	#forward propagation
	z1 = input_layer.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	a = np.absolute(z2);
	z2 = z2/np.sum(a,axis=1,keepdims = True)
	pr = np.exp(z2)
	c = 0

	div = (np.sum(pr,axis=1,keepdims = True))


	prob = pr/div
	np.savetxt("outputkaggle.csv", np.argmax(prob,axis = 1), delimiter=",")



def buildmodel( hidden_layers, input_layer , iterations):
	output_layer_size = np.unique(df['label']).size
	print output_layer_size
	#shape of output will be (train_size,output_layer_size)
	hidden_layers = 2
	layer1_size = 500
	input_layer_size = pixel_cnt 
	W2 = np.random.randn(layer1_size,output_layer_size)/np.sqrt(layer1_size)
	b2 = np.zeros((1,output_layer_size)) 	
	W1 = np.random.randn(input_layer_size,layer1_size)/np.sqrt(input_layer_size)
	b1 = np.zeros((1,layer1_size))
	for i in range(iterations):
		
		#first apply forward propagation for every iterations
		z1 = input_layer.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		
		#using softmax classifier
		#keepdims is for keeping the dimensions retained
		a = np.absolute(z2);
		z2 = z2/np.sum(a,axis = 1 , keepdims = True)
		pr = np.exp(z2)
		div = (np.sum(pr,axis=1,keepdims = True))
		
		prob = pr/div
		#now apply backpropagation algorithm
		delta3 = prob
		print "Iteration " + str(i) 
		delta3[range(train_size),output]-=1
	
		# dlW3 = (a2.T).dot(delta4)
		# dlB3 = np.sum(delta4,axis = 0 , keepdims = True)

		# delta3 = (delta4.dot(W3.T))*(1 - np.power(a2,2))
		dlW2 = (a1.T).dot(delta3)
		dlB2 = np.sum(delta3,axis=0 ,keepdims = True)

		delta2 = (delta3.dot(W2.T))*(1 - np.power(a1,2))
		dlW1 = (input_layer.T).dot(delta2)
		dlB1 = np.sum(delta2,axis=0 ,keepdims = True)

		dlW1 += lmbda*W1
		dlW2 += lmbda*W2

		W1 -= gr_descent*dlW1
		W2 -= gr_descent*dlW2
		b1 -= gr_descent*dlB1
		b2 -= gr_descent*dlB2


		
	model = {'W1':W1 , 'W2':W2 , 'b1':b1 , 'b2':b2  }
	return model

model = buildmodel(2,train_data,60)
print model['W1'];
ar = test_data(model)