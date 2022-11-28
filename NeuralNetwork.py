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
neurons = 8
input_size = 2
learning_rate = 0.3


def parse_inputs(csv): 
    str = f'public/{csv}'
    return genfromtxt(str, delimiter=',')

# layer by layer oeprations 
# number of times calculator is dependent upon how many layers i have 
# size of wieght matrix changes based how many neurons in a layer 
def d_activation_tanh(input_val, prev_deriv):
    return (1.-np.tanh(input_val)**2) * prev_deriv

def activation_tanh(input_val):
    return np.tanh(input_val)

def activation_sigmoid(input_val):
    #return np.softmaz()
    #for item in input_val:
    return 1/(1 + np.exp(-input_val))

# x is input before 
# y 
# fdor last layer is deriv loss function (matrix)
# for all other layers deriv = input @ prev deriv #        backprop(layer4_data[1], 0, layer4_data[0], d_loss_mat )

def d_activation_sigmoid(x, prev_deriv):
    val = activation_sigmoid(x) * (1 - activation_sigmoid(x))
    val *= prev_deriv
    return val

# calc all wieghts at once and feed back
def backprop(weights, bias, input, act, der_loss, label): 
    m = input.shape[1]
    input = input.T
    #temp = act - label.T
    #d_act_matrix = d_activation_sigmoid() ## FIX later
    dw = (1 / m) * np.dot(input, der_loss) # derloss instead if not wokring 
    # db = (1 / m) * np.sum( act - label )
    weights -= dw * learning_rate 
    return  der_loss @ weights.T # test shape
    #return weights 

def loss(label, output): # cost is matrix
    # batch size
    label = label.reshape(-1,1)
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))

    print(cost)
    return loss 

def back_loss(label ,output ): # d_csorss * w.T
    # loss = expected_label - predicted 
    d_cross = ((label/output) - (1-label)/(1 - output))
    print(d_cross)
    return d_cross
    
def perceptron(input1, layer_num): 
    # batch one size 3
    print("\n\nLayer Num is : ", layer_num)
    
    if (layer_num > 1):
        weights = np.random.normal(size = (neurons, neurons))
    elif (layer_num % 2 ) != 0:
        input1 = input1[:batch_size]
        weights = np.random.normal(size = (input_size, neurons))
    
    print("Input: ", np.shape(input1) )
    print("Weights: ",np.shape( weights) )
    output = np.dot( input1, weights ) + bias #np.matmul(input1, weights )
    print("Post matrix mul: ",np.shape( (output)) )

    if layer_num == 4: 
        output = activation_sigmoid(output)
        print("LAYER 4")
    else: output = activation_tanh(output) 
    print("after tan ", np.shape(output))
    
    return [output, weights, input1]


if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv

    counter = 0
    for i in sys.argv:
        if counter == 1:    train_data = parse_inputs(i)
        elif counter == 2:  train_label = parse_inputs(i)
        elif counter == 3:  test_data = parse_inputs(i)
        elif counter == 4:   test_label = parse_inputs(i)
        counter += 1


    print(int((len(train_data))/ batch_size))
    for j in range( int(int(len(train_data))/ batch_size)):
        # parse datum for minibatch sizes 
        train_data_new = train_data[j*batch_size : (j+1)*batch_size]
        train_label_new = train_label[j*batch_size : (j+1)*batch_size]

        layer1_data = perceptron(train_data_new, layer_num = 1)
        weights1 = layer1_data[1]
        layer2_data = perceptron(layer1_data[0], layer_num = 2)
        weights2 = layer2_data[1]
        layer3_data = perceptron(layer2_data[0], layer_num = 3)
        weights3 = layer3_data[1]
        layer4_data = perceptron(layer3_data[0], layer_num = 4)
        weights4 = layer4_data[1]

        print(np.shape(layer4_data[0])) 
        avg_loss = loss(train_label_new, layer4_data[0])
        d_loss_mat = back_loss(layer3_data[0], layer4_data[0])
        d_sig = d_activation_sigmoid(d_loss_mat,layer4_data[2]) # FUNCTION 
        bp_lay4 =  backprop(layer4_data[1], 0, layer3_data[0], layer4_data[0], d_sig, train_label_new )

        d_tan = d_activation_tanh(d_loss_mat, layer3_data[0])

        #backprop(weights, bias, input, act, der_loss, label):

        """dbp_tan3 = d_tanh()
        bp_lay4 = """
        # store weigths in return statement and update after first epoc to layer day 

        #weights, bias, input, act, der_loss
                    # weights       
        # D-loss 
