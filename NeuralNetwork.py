import numpy as np
import sys
import argparse as parser
import csv
import random, os
from numpy import genfromtxt

lr = .3 #learning rate
bias = 0 #value of bias
epoch = 300
batch_size = 30
# weights = [random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
# weights = np.random.rand(batch_size,batch_size)
neurons = 3
input_size = 2

"""layers = 3
 
weights = [100, 0.002 , 12]
x1 = [-0.2, 1.1, 4.9]
x2 = [0.2, -3.0, 1.0]"""


def parse_inputs(csv): 
    str = f'public/{csv}'
    return genfromtxt(str, delimiter=',')

#forward pass 
"""2,3     3,3
    (inputsize, batchsize) (batchsize,batchsize)

[   [-0.2, 1.1, 4.9]
    [0.2, -3.0, 1.0]    ]  * [ [100, 0.002 , 12]
                                [100, 0.002 , 12]
                                [100, 0.002 , 12]
                                                    ]   = 2x3 []""" # transpose till works


# layer by layer oeprations 
# number of times calculator is dependent upon how many layers i have 
# size of wieght matrix changes based how many neurons in a layer 
      
def activation_tanh(input_val):
    return np.tanh(input_val)

def activation_sigmoid(input_val):
    #return np.softmaz()
    #for item in input_val:
    return 1/(1 + np.exp(-input_val))
   
    
def perceptron(input1, layer_num): 
    # batch one size 3
    print("\n\nLayer Num is : ", layer_num)

    # M x N
    #if layer_num == 1 :         #N     x   K 
    if layer_num == 3 :
        input1 = input1[:batch_size].T
        weights = np.random.rand(neurons,input_size)
    if layer_num == 4 :
        input1 = input1[:batch_size]
        weights = np.random.rand(neurons,input_size).T
    elif (layer_num % 2) == 0: 
        input1 = input1[:batch_size]
        weights = np.random.rand(neurons,input_size).T
    
    elif (layer_num % 2 ) != 0:
        input1 = input1[:batch_size].T
        #print(np.shape("ORG size" , input1))
        weights = np.random.rand(neurons,input_size)
    
    print("Input: ", np.shape(input1) )
    print("Weights: ",np.shape( weights) )
    # print(np.shape(weights), np.shape(input1))
    # output = np.matmul(input1, weights)
    output = np.dot( weights, input1) + bias #np.matmul(input1, weights )
    print("Post matrix mul: ",np.shape( (output)) )
    #if layer_num == 1 : 

    output = activation_tanh(output) 
    #elif layer_num == 2: 
     #   output = ac
    print("after tan ", np.shape(output))
    
    return [output, weights]

def loss_function(output_val, label):
    loss = []
    for i in range(batch_size):
        print("\n" , output_val[0][i])
        print(label[i])
        loss.append( (label[i] - output_val[0][i]) ** 2)
        print(loss[i])
    print(loss)
        
if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv

    counter = 0
    for i in sys.argv:
        if counter == 1 :    train_data = parse_inputs(i)
        elif counter == 2 :  train_label = parse_inputs(i)
        elif counter == 3 :  test_data = parse_inputs(i)
        elif counter == 4:   test_label = parse_inputs(i)
        counter += 1
    
    
    """print(type(train_data))
    print(np.shape(train_label))
    print(np.shape(test_data))"""
    layer1_data = perceptron(train_data, layer_num = 1)
    weights1 = layer1_data[1]
    layer2_data = perceptron(layer1_data[0], layer_num = 2)
    weights2 = layer2_data[1]
    layer3_data = perceptron(layer2_data[0], layer_num = 3)
    weights3 = layer3_data[1]
    layer4_data = perceptron(layer3_data[0], layer_num = 4)
    weights4 = layer4_data[1]

    print(np.shape(layer4_data[0])) 
    loss_function(layer4_data[0], train_label)

    for i in layer4_data[0]:
        print(i)

    """for i in range(epoch):
        # call prepcrotron and return new datum 
        nothing = 0
    """
"""def Perceptron(input1, input2, output) :
   outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
   if outputP > 0 : #activation function (here Heaviside)
      outputP = 1
   else :
      outputP = 0
   error = output - outputP
   weights[0] += error * input1 * lr
   weights[1] += error * input2 * lr
   weights[2] += error * bias * lr"""