import numpy as np
import sys
import argparse as parser
import csv
import random, os
from numpy import genfromtxt

bias = 0 #value of bias
epochs = 300
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
    val = 1/(1 + np.exp(-input_val))
    #print(" val :", val)

    return val
    


# x is input before 
# y 
# fdor last layer is deriv loss function (matrix)
# for all other layers deriv = input @ prev deriv #        backprop(layer4_data[1], 0, layer4_data[0], d_loss_mat )

def d_activation_sigmoid(x, prev_deriv):
    val = activation_sigmoid(x) * (1 - activation_sigmoid(x))
    val *= prev_deriv
    return val

# calc all wieghts at once and feed back
def backprop(weights, input, der_loss): 
    m = input.shape[1]
    input = input.T
    #temp = act - label.T
    #d_act_matrix = d_activation_sigmoid() ## FIX later
    dw = (1 / m) * np.dot(input, der_loss) # derloss instead if not wokring 
    # db = (1 / m) * np.sum( act - label )
    weights -= dw * learning_rate 
    return [der_loss @ weights.T, weights]# test shape
    #return weights 


def loss(label, output): # cost is matrix
    # batch size
    label = label.reshape(-1,1)
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))
    return cost 


def back_loss(label ,output ): # d_csorss * w.T
    # loss = expected_label - predicted 
    d_cross = ((label/output) - (1-label)/(1 - output))
    #print(d_cross)
    return d_cross
    

def forward_pass(input1, layer_num, weights, iteration): 
    # batch one size 3
    # print("\n\nLayer Num is : ", layer_num)
    
    if iteration == 0:
        #print("iteration is : ",iteration)
        if (layer_num > 1): weights = np.random.normal(size = (neurons, neurons))
        elif (layer_num % 2 ) != 0: weights = np.random.normal(size = (input_size, neurons))

    #print("Weights: ",np.shape( weights), "layer num: ", layer_num )
    output = np.dot( input1, weights ) + bias #np.matmul(input1, weights )


    if layer_num == 4: 
        output = activation_sigmoid(output)

    else: output = activation_tanh(output) 
    return [output, weights, input1]


if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv

    weights1, weights2, weights3, weights4, counter = 0, 0, 0, 0, 0
    
    for i in sys.argv:
        if counter == 1:    train_data = parse_inputs(i)
        elif counter == 2:  train_label = parse_inputs(i)
        elif counter == 3:  test_data = parse_inputs(i)
        elif counter == 4:   test_label = parse_inputs(i)
        counter += 1
    
    for x in range(epochs):
    #print(int((len(train_data))/ batch_size))
        for j in range( int(int(len(train_data))/ batch_size)):
            # parse datum for minibatch sizes 
            train_data_new = train_data[j*batch_size : (j+1)*batch_size]
            train_label_new = train_label[j*batch_size : (j+1)*batch_size]

            if j == 0:
                layer1_data = forward_pass(train_data_new, layer_num = 1, weights = None, iteration = j)
                weights1 = layer1_data[1]
                #print("forward pass weigths shape: ", np.shape(weights1))

                layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = None, iteration = j)
                weights2 = layer2_data[1]
                #print("forward pass weigths shape: ", np.shape(weights2))

                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = None, iteration = j)
                weights3 = layer3_data[1]
                #print("forward pass weigths shape: ", np.shape(weights3))

                layer4_data = forward_pass(layer3_data[0], layer_num = 4, weights = None, iteration = j)
                weights4 = layer4_data[1]
                #print("forward pass weigths shape: ", np.shape(weights4))
            else: # use weights from first iteration 
                layer1_data = forward_pass(train_data_new, layer_num = 1, weights = weights1, iteration = j)
                weights1 = layer1_data[1]
                layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = weights2, iteration = j)
                weights2 = layer2_data[1]
                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = weights3, iteration = j)
                weights3 = layer3_data[1]
                layer4_data = forward_pass(layer3_data[0], layer_num = 4, weights = weights4, iteration = j)
                weights4 = layer4_data[1]
            if x == 299:
                print(layer4_data)
                layer4_data = np.round(layer4_data)
                break 
            #if j != 0: 
            temp_w = weights4
            #print(np.shape(layer4_data[0])) 
            avg_loss = loss(train_label_new, layer4_data[0])
            d_loss_mat = back_loss(layer3_data[0], layer4_data[0])

            d_sig = d_activation_sigmoid(d_loss_mat,layer4_data[2]) # FUNCTION 
            bp_lay4 =  backprop(layer4_data[1], layer3_data[0], d_sig)
            #print("BP LAYER 4: ",  bp_lay4[1].shape)
            weights4 = bp_lay4[1]

            assert(temp_w != weights4, f"Weights are te same size {temp_w.shape, weights4.shape}")

            #print("BP LAYER 4: ", weights4.shape)

            d_tan3 = d_activation_tanh(d_loss_mat, layer3_data[0])
            d_tan3 *= bp_lay4[0]
            bp_lay3 = backprop(layer3_data[1], layer2_data[0], d_tan3)
            weights3 = bp_lay3[1]
            #print(bp_lay3[0].shape)

            d_tan2 = d_activation_tanh(d_loss_mat, layer2_data[0])
            d_tan2 *= bp_lay3[0]
            bp_lay2 = backprop(layer2_data[1], layer1_data[0], d_tan2)
            weights2 = bp_lay2[1]
            #print(bp_lay2[0].shape)

            d_tan1 = d_activation_tanh(d_loss_mat, layer1_data[0])
            d_tan1 *= bp_lay2[0]
            bp_lay1 = backprop(layer1_data[1], train_data_new, d_tan1)
            weights1 = bp_lay1[1]
            #print("BP LAYER1: ", bp_lay1[1].shape)
            #print("Weights layer 1: ", weights1.shape)


            # store weigths in return statement and update after first epoc to layer day 

            #weights, bias, input, act, der_loss
                        # weights       
            # D-loss 
    
