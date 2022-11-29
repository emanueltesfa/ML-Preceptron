import numpy as np
import sys
import argparse as parser
import csv
import random, os
from numpy import genfromtxt
import time
#import matplotlib.pyplot as plt


#bias = 0 #value of bias
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
    return (1 -(np.tanh(input_val)**2)) * prev_deriv


def activation_tanh(input_val):
    return np.tanh(input_val)


def activation_sigmoid(input_val):
    #return np.softmaz()
    #for item in input_val:
    val = 1/(1 + np.exp(-1*input_val))
    #print(" val :", val)

    return val


# x is input before 
# y 
# fdor last layer is deriv loss function (matrix)
# for all other layers deriv = input @ prev deriv #        backprop(layer4_data[1], 0, layer4_data[0], d_loss_mat )

def d_activation_sigmoid(prev_deriv, x):
    val = activation_sigmoid(x) * (1 - activation_sigmoid(x))
    #print("VAL SHPE" , prev_deriv.shape)
    val *= prev_deriv
    return val

# calc all wieghts at once and feed back
def backprop(weights, input, der_loss ,prev_bias, layer_num): 
    #m = input.shape[1]
    input = input.T
    #print("\n\nInput backprop: " , input.shape, "der loss: ", der_loss.shape)
    dw = np.dot(input.reshape(-1, 1), der_loss.reshape(1, -1) ) # derloss instead if not wokring
    #print( "dw,", dw)
    db = der_loss
    prev_bias = prev_bias - (db * learning_rate)
    #print("db", db.shape)
    weights = weights - ( dw * learning_rate )
    return [der_loss @ weights.T, weights, prev_bias ]# test shape

def leaky_Relu(x):
    #print("leaky")
    return np.where(x > 0, x, x * 0.01)                          

def d_leaky_Relu(x, alpha=0.01):
    return np.where(x>0, 1, alpha) 

def accuracy(label, pred, epoch):
    counter = 0
    pred = np.array(pred)
    #print(label.shape)
    #print(pred)
    pred = np.round(pred)
    
    for i in range(len(label)) : 
        if pred[i] >= 0:
            pred[i] = 1.
        else: 
            pred[i] = -1.
        #print(pred[i], label[i])
        if label[i] == pred[i]:
            counter += 1 
    print("Num corect predictions is: ", counter)


def loss(label, output): # cost is matrix
    #print("label ", label, "OUTPUT  ", output)
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))

    return cost 


def back_loss(label ,output ): # d_csorss * w.T
    label = label.reshape(1, 1)
    #print("label.shape: ", label.shape, "output:", output.shape )

    d_cross = ((label/output) - (1-label)/(1 - output))
    #print(d_cross.shape)
    return d_cross
    
def d_mean_squared_loss(label, output_val):
    return (output_val - label)


# output network - excpted 
def forward_pass(input1, bias, layer_num,  weights, iteration): 
    # batch one size 3
    output = None
    if iteration == 0:
        #print("iteration is : ",iteration)
        if layer_num == 4: 
            weights = np.random.normal(size = (2, 1))
            print("WEIGHTS UPDATED")
        elif (layer_num > 1): 
            weights = np.random.normal(size = (neurons, neurons))
            print("WEIGHTS UPDATED")

        elif (layer_num % 2 ) != 0:
            weights = np.random.normal(size = (input_size, neurons))
            print("WEIGHTS UPDATED")


    #print("Weights: ",np.shape( weights), "layer num: ", layer_num )
   
    #print("bias is: ", bias)
    output_temp = np.dot( input1, weights ) + bias  #np.matmul(input1, weights )
    #print("output aftre dot, ", output_temp)
    if layer_num == 4:
        output = activation_tanh(output_temp)

    else: output = activation_tanh(output_temp) 
    #print("weights in forward, ", weights)
    #print("output after relu, ", output)

    return [output, weights, output_temp, input1]

def tanh_fix(label):
    if label == 0: return -1
    else: return 1

if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv
    # python3 NeuralNetwork.py gaussian_train_data.csv gaussian_train_label.csv gaussian_test_data.csv


    weights1, weights2, weights3, weights4, counter =  0, 0, 0, 0, 0
    bias4, bias3, bias2, bias1 = 0.1, 0.1, 0.1, 0.1

    for i in sys.argv:
        if counter == 1: train_data = parse_inputs(i)
        elif counter == 2: train_label = parse_inputs(i)
        elif counter == 3: test_data = parse_inputs(i)
        elif counter == 4: test_label = parse_inputs(i)
        counter += 1
    #print(train_label)
    for x in range(epochs):
    #print(int((len(train_data))/ batch_size))
        total_loss = 0
        final_output_mat = []
        final_label_mat = []
        time.sleep(1)
        for j in range(int(int(len(train_data))/ batch_size)):
            #print(j)
            # parse datum for minibatch sizes 
            train_data_new = train_data[j*batch_size : (j+1)*batch_size]
            train_label_new = train_label[j*batch_size : (j+1)*batch_size]
            #print("train label old", train_label_new)
            
            train_label_new = tanh_fix(train_label_new)
            #print("train label new", train_label_new)

            #train_data_new = train_data[0]
            #train_label_new = train_label[0]
            #print(train_label[0])
            #print("Input data: ", train_data_new)


            if j == 0 and x == 0:
                """ layer1_data = forward_pass(train_data_new, layer_num = 1, weights = None, iteration = x+j)
                weights1 = layer1_data[1]"""

                """layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = None, iteration = x+j)
                weights2 = layer2_data[1]

                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = None, iteration = x+j)
                weights3 = layer3_data[1]"""

                layer4_data = forward_pass(train_data_new, bias4, layer_num = 4, weights = None, iteration = x+j)
                weights4 = layer4_data[1]
            else: # use weights from first iteration 
                """layer1_data = forward_pass(train_data_new, layer_num = 1, weights = weights1, iteration = x+j)
                weights1 = layer1_data[1]"""
                """layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = weights2, iteration = x+j)
                weights2 = layer2_data[1]
                layer3_data = forward_pass(layer2_data[0], layer_num = 3, weights = weights3, iteration = x+j)
                weights3 = layer3_data[1]"""

                layer4_data = forward_pass(train_data_new, bias4, layer_num = 4, weights = weights4, iteration = x+j)
                weights4 = layer4_data[1]

            #print("layer4_data[0][0]", layer4_data[0][0])
            # print(lay)
            final_output = layer4_data[0][0]
            #print(layer4_data[0][0])
            final_output_mat.append(final_output[0])
            final_label_mat.append(train_label_new)
            
            #total_loss += mean_squared_loss(train_label_new, final_output)
            #accuracy(train_label_new, final_output_mat ,epoch = x)
            #print(j)
            #print("loss calc: ", total_loss)
            if j == 2000 : 
                #print(np.array(final_output_mat).shape)

                print("epoch", x)
                #print(total_loss)
                #print(np.array(final_label_mat).shape)
                print("output, ", final_output_mat)
                print("label", final_label_mat)
                accuracy(final_label_mat, final_output_mat ,epoch = x)
            #print("total_loss: ", total_loss)
            #if j != 0: 
            temp_w = weights4

            #d_loss_mat = back_loss(train_label_new, layer4_data[0])
            d_loss_mat = d_mean_squared_loss(train_label_new, layer4_data[0])
            #print("d_loss from mean squ: ", d_loss_mat )
            d_sig = d_activation_tanh(layer4_data[2] ,d_loss_mat) # FUNCTION 
           # print("d_sig: ", d_sig)

            #print("bias4 before updated " ,bias4)
            bp_lay4 =  backprop(layer4_data[1], layer4_data[3], d_sig, bias4, 4)
            weights4 = bp_lay4[1]
            bias4 = bp_lay4[2] 
            #print("bias4 after update, " ,bias4)

            #print(temp_w, weights4)
            #assert( np.alltrue(temp_w == weights4) )



            """d_tan3 = d_leaky_Relu(layer3_data[2]) * bp_lay4[0] # dLossmat => bp_lay4 change
            d_tan3 *= bp_lay4[0]
            bp_lay3 = backprop(layer3_data[1], layer2_data[0], d_tan3, 3)
            weights3 = bp_lay3[1]
            #print(bp_lay3[0].shape)

            d_tan2 = d_leaky_Relu( layer2_data[2]) *  bp_lay3[0]  # dLossmat => bp_lay3 change
            d_tan2 *= bp_lay3[0]
            bp_lay2 = backprop(layer2_data[1], layer1_data[0], d_tan2, 2)
            weights2 = bp_lay2[1]
            #print(bp_lay2[0].shape)"""

            #d_tan1 = d_leaky_Relu(layer1_data[2]) * bp_lay2[0]  # dLossmat => bp_lay2 change
            #d_tan1 *= bp_lay2[0]
            """bp_lay1 = backprop(layer1_data[1], train_data_new, d_sig,1)
            weights1 = bp_lay1[1]"""
            #print("epoch: ",x, "batch num: " ,j, "After backprop weights", weights4, " output: ", layer4_data[0], "exptected: ", train_label_new)
            
  