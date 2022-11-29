import numpy as np
import sys
import argparse as parser
import csv
import random, os
from numpy import genfromtxt
import time
#import matplotlib.pyplot as plt


bias = 0 #value of bias
epochs = 300
batch_size = 1
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
    #print("VAL SHPE" , prev_deriv.shape)

    val *= prev_deriv

    return val

# calc all wieghts at once and feed back
def backprop(weights, input, der_loss, layer_num): 
    m = input.shape[1]
    input = input.T
    #der_loss = der_loss
    #temp = act - label.T
    #d_act_matrix = d_activation_sigmoid() ## FIX later
    #print("Input shape: and derLoss " , input.shape, der_loss.shape)
    #print("der_loss: ", der_loss)
    dw = np.dot(input.reshape(1, -1).T, der_loss.reshape(-1, 1).T ) # derloss instead if not wokring 
    # db = (1 / m) * np.sum( act - label )
    #print("weights shape prior: ", weights.shape, "Dw: ", dw.shape)

    weights = weights - ( dw * learning_rate )
    #print("Weights shape: " , weights.shape)
    return [der_loss @ weights.T, weights]# test shape
    #return weights 

def accuracy(label, pred, epoch):
    counter = 0
    pred = np.array(pred)
    #print(label.shape)
    #print(pred)
    pred = np.round(pred)
    for i in range(len(label)) : 
        if label[i] == pred[i]:
            counter += 1 
    print("Num corect predictions is: ", counter/2001)


def loss(label, output): # cost is matrix
    # batch size
    #label = label.reshape(-1,1)
    #print("label ", label, "OUTPUT  ", output)
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))

    return cost 


def back_loss(label ,output ): # d_csorss * w.T
    # loss = expected_label - predicted 
    #print("label.shape: ", label.shape, "output:", output.shape )
    d_cross = ((label/output) - (1-label)/(1 - output))
    
    #print(d_cross.shape)
    return d_cross
    

def forward_pass(input1, layer_num, weights, iteration): 
    # batch one size 3
    output = None
    if iteration == 0:
        #print("iteration is : ",iteration)
        if layer_num == 4: 
            weights = np.random.normal(size = (neurons, 1))
            print("WEIGHTS UPDATED")
        elif (layer_num > 1): 
            weights = np.random.normal(size = (neurons, neurons))
            print("WEIGHTS UPDATED")

        elif (layer_num % 2 ) != 0:
            weights = np.random.normal(size = (input_size, neurons))
            print("WEIGHTS UPDATED")


    #print("Weights: ",np.shape( weights), "layer num: ", layer_num )
    output_temp = np.dot( input1, weights ) + bias #np.matmul(input1, weights )

    if layer_num == 4:
        output = activation_sigmoid(output_temp)

    else: output = activation_tanh(output_temp) 
    return [output, weights, output_temp]


if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv

    weights1, weights2, weights3, weights4, counter =  0, 0, 0, 0, 0
    
    for i in sys.argv:
        if counter == 1: train_data = parse_inputs(i)
        elif counter == 2: train_label = parse_inputs(i)
        elif counter == 3: test_data = parse_inputs(i)
        elif counter == 4: test_label = parse_inputs(i)
        counter += 1
    
    for x in range(epochs):
    #print(int((len(train_data))/ batch_size))
        total_loss = 0
        final_output_mat = []
        time.sleep(1)
        for j in range( int(int(len(train_data))/ batch_size)):
            #print(j)
            # parse datum for minibatch sizes 
            train_data_new = train_data[j*batch_size : (j+1)*batch_size]
            train_label_new = train_label[j*batch_size : (j+1)*batch_size]

            if j == 0 and x == 0:
                layer1_data = forward_pass(train_data_new, layer_num = 1, weights = None, iteration = x+j)
                weights1 = layer1_data[1]
                #print("forward pass weigths shape: ", np.shape(weights1))

                layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = None, iteration = x+j)
                weights2 = layer2_data[1]
                #print("forward pass weigths shape: ", np.shape(weights2))

                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = None, iteration = x+j)
                weights3 = layer3_data[1]
                #print("forward pass weigths shape: ", np.shape(weights3))

                layer4_data = forward_pass(layer3_data[0], layer_num = 4, weights = None, iteration = x+j)
                weights4 = layer4_data[1]
                #print("Output",layer4_data[0].shape, "weifhts: ", weights4.shape)

                #print("forward pass weigths shape: ", np.shape(weights4))
            else: # use weights from first iteration 
                layer1_data = forward_pass(train_data_new, layer_num = 1, weights = weights1, iteration = x+j)
                weights1 = layer1_data[1]
                layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = weights2, iteration = x+j)
                weights2 = layer2_data[1]
                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = weights3, iteration = x+j)
                weights3 = layer3_data[1]
                layer4_data = forward_pass(layer3_data[0], layer_num = 4, weights = weights4, iteration = x+j)
                weights4 = layer4_data[1]
            #if x == 299:
            #layer4_data = np.round(layer4_data)
            
            #print(j, x)
            final_output = layer4_data[0][0]
            final_output_mat.append(final_output)
            
            #print(final_output)
            #final_output = np.round(final_output)
            total_loss += loss(train_label[j], final_output)
            
            #print(final_output)
            if j == 2000 and x ==299: 
                print(x)
                print(total_loss)
                accuracy(train_label, final_output_mat ,epoch = x)
                 
            #if j != 0: 
            temp_w = weights4
            #print(np.shape(layer4_data[0])) 
            #avg_loss = loss(train_label_new, layer4_data[0])
            d_loss_mat = back_loss(train_label_new, layer4_data[0])
            #print("DLOSS, ", d_loss_mat.shape)
            #print("input lay 4: ", layer4_data[2].shape)

            d_sig = d_activation_sigmoid(d_loss_mat, layer4_data[2]) # FUNCTION 
            bp_lay4 =  backprop(layer4_data[1], layer3_data[0], d_sig, 4)
            #print("BP LAYER 4: ",  bp_lay4[1].shape)
            weights4 = bp_lay4[1]

            #print(temp_w, weights4)
            #assert( np.alltrue(temp_w == weights4) )

            #print("BP LAYER 4: ", weights4.shape)

            d_tan3 = d_activation_tanh(d_loss_mat, layer3_data[0])
            d_tan3 *= bp_lay4[0]
            bp_lay3 = backprop(layer3_data[1], layer2_data[0], d_tan3, 3)
            weights3 = bp_lay3[1]
            #print(bp_lay3[0].shape)

            d_tan2 = d_activation_tanh(d_loss_mat, layer2_data[0])
            d_tan2 *= bp_lay3[0]
            bp_lay2 = backprop(layer2_data[1], layer1_data[0], d_tan2, 2)
            weights2 = bp_lay2[1]
            #print(bp_lay2[0].shape)

            d_tan1 = d_activation_tanh(d_loss_mat, layer1_data[0])
            d_tan1 *= bp_lay2[0]
            bp_lay1 = backprop(layer1_data[1], train_data_new, d_tan1,1)
            weights1 = bp_lay1[1]
            #print("epoch: ",x, "batch num: " ,j, "Weifht", weights1[0][0])
  