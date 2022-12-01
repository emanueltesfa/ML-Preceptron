import numpy as np
import sys
import argparse as parser
import csv
import random
import os
from numpy import genfromtxt
import time
import copy
# import matplotlib.pyplot as plt

epochs = 1500
batch_size = 20
neurons = 8
input_size = 2
learning_rate = 0.1


def parse_inputs(csv):
    str = f'public/{csv}'
    return genfromtxt(str, delimiter=',')


def d_activation_tanh(input_val, prev_deriv):
    return (1 - (np.tanh(input_val)**2)) * prev_deriv


def activation_tanh(input_val):
    return np.tanh(input_val)


def activation_sigmoid(input_val):
    val = 1/(1 + np.exp(-1*input_val))
    return val


# x is input before
# y
# fdor last layer is deriv loss function (matrix)
# for all other layers deriv = input @ prev deriv #        backprop(layer4_data[1], 0, layer4_data[0], d_loss_mat )

def d_activation_sigmoid(prev_deriv, x):
    val = activation_sigmoid(x) * (1 - activation_sigmoid(x))
    val *= prev_deriv
    return val


# calc all wieghts at once and feed back
def backprop(weights, input, der_loss, prev_bias, layer_num):
    input = input.T
    dw = np.dot(input.reshape(-1, 1), der_loss.reshape(1, -1))
    db = der_loss

    return [der_loss @ weights.T, dw, db]  # test shape



def set_weights_bias(t_b3, b3, t_b4, b4, t_w3, w3, t_w4,  w4, new_batch_size):
    b3 = b3 - (t_b3 * learning_rate) / new_batch_size
    b4 -= (t_b4 * learning_rate) / new_batch_size
    w3 -= (t_w3 * learning_rate) / new_batch_size
    w4 -= (t_w4 * learning_rate) / new_batch_size

    return b3, b4, w3, w4


def leaky_Relu(x):
    new_arr = copy.deepcopy(x)
    new_arr = new_arr.reshape(-1, 1)

    for i in range(len(new_arr)):
        if new_arr[i] < 0:
            new_arr[i] *= 0.01
    return new_arr.T
    


def d_leaky_Relu(old_x, alpha=0.01):
    x = copy.deepcopy(old_x)
    x = x.reshape(-1, 1)

    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.01
        else:
            x[i] = 1
    return x.T


def accuracy(label, pred, epoch):
    counter = 0
    pred = np.array(pred)

    for i in range(len(pred)):
        if pred[i] >= 0:
            pred[i] = 1.
        else:
            pred[i] = -1.
        # print(pred[i], label[i])
        if label[i] == pred[i]:
            counter += 1
    print("Num correct predictions is: ", counter)


def accuracy2(label, pred, epoch):
    counter = 0
    pred = np.array(pred)

    for i in range(len(pred)):
        if label[i] == pred[i]:
            counter += 1
    print("Num correct predictions is: ", counter)


def loss(label, output):  # cost is matrix
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))
    return cost


def back_loss(label, output):  # d_csorss * w.T
    label = label.reshape(1, 1)
    d_cross = ((label/output) - (1-label)/(1 - output))
    return d_cross


def d_mean_squared_loss(label, output_val):
    return (output_val - label)


def forward_pass(input1, bias, layer_num,  weights, iteration):

    output = None
    output_temp = np.dot(input1, weights) + bias  # np.matmul(input1, weights )
    output = activation_tanh(output_temp)
    return [output, weights, output_temp, input1]

def set_inputs(inp):
    #for ix, iy in np.ndindex(inp.shape):
    return np.vstack(( inp[:, 0],  inp[:, 1],  inp[:, 0] ** 2,  inp[:, 1] ** 2,  inp[:, 0] * inp[:, 1], np.sin(inp[:, 0]),  np.sin(inp[:, 1]) ) ).T
                    #   1               2           3               4                      6                  7                       8
def tanh_fix(label):
    for i in range(len(label)):
        # print(label[i])
        if label[i] == 0:
            label[i] = -1
        else:
            label[i] = 1
    return label


if __name__ == "__main__":
    # python3 NeuralNetwork.py xor_train_data.csv xor_train_label.csv xor_test_data.csv
    # python3 NeuralNetwork.py gaussian_train_data.csv gaussian_train_label.csv gaussian_test_data.csv
    # python3 NeuralNetwork.py spiral_train_data.csv spiral_train_label.csv spiral_test_data.csv spiral_test_data.csv                                                                                                               

    start = time.time()
    weights1, weights2, weights3, weights4, counter = 0, 0, 0, 0, 0
    bias3, bias4 = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(1, 8), 0.1

    t_weights3, t_weights4 = np.zeros((2, 8)), np.zeros((8, 1))
    t_bias4, t_bias3 = np.zeros((1, 1)), np.zeros((1, 8))

    for i in sys.argv:
        if counter == 1:
            train_data = parse_inputs(i)
        elif counter == 2:
            train_label = parse_inputs(i)
        elif counter == 3:
            test_data = parse_inputs(i)
        elif counter == 4:
            test_label = parse_inputs(i)
        counter += 1

    train_data = set_inputs(train_data)
    test_data = set_inputs(test_data)

    weights4 = np.random.normal(size=(neurons, 1))
    weights3 = np.random.normal(size=(7, neurons))

    train_label = tanh_fix(train_label)

    for x in range(epochs):
        total_loss = 0
        final_output_mat = []
        final_label_mat = []
        # time.sleep(1)

        for j in range(int(int(len(train_data)) / batch_size)):
            # print("Epoch is is: ", x)
            train_data_new = train_data[j*batch_size: (j+1)*batch_size]
            train_label_new = train_label[j*batch_size: (j+1)*batch_size]
            new_batch_size = len(train_data_new)
 
            for i in range(len(train_data_new)):  # iterate items in batch
                
                # FOWARD PASS
                layer3_data = forward_pass(train_data_new[i], bias3, layer_num=3, weights=weights3, iteration=x+j)
                layer4_data = forward_pass(layer3_data[0], bias4, layer_num=4, weights=weights4, iteration=x+j)


                final_output = layer4_data[0][0]
                final_output_mat.append(final_output[0])
                final_label_mat.append(train_label_new[i])


                d_loss_mat = d_mean_squared_loss(
                    train_label_new[i], layer4_data[0])

                #LAYER 4 BACKRPROP
                d_tan4 = d_activation_tanh(
                    layer4_data[2], d_loss_mat)  # FUNCTION
                bp_lay4 = backprop(weights4, layer4_data[3], d_tan4, bias4, 4)
                t_weights4 = bp_lay4[1]
                t_bias4 = bp_lay4[2]
           
                # assert( np.alltrue(temp_w == weights4) )

                #LAYER 3 BACKRPROP Chain deriv using d_tan
                d_leaky3 = d_activation_tanh(layer3_data[2], bp_lay4[0])
                bp_lay3 = backprop(
                    weights3, layer3_data[3], d_leaky3, bias3, 3)
                t_weights3 = bp_lay3[1]
                t_bias3 = bp_lay3[2]

            output_bias_weights = set_weights_bias(t_bias3, bias3, t_bias4, bias4, t_weights3, weights3,
                                                   t_weights4, weights4, new_batch_size)
            bias3 = output_bias_weights[0]
            bias4 = output_bias_weights[1]
            weights3 = output_bias_weights[2]
            weights4 = output_bias_weights[3]

        accuracy(train_label, final_output_mat, epoch=x)

    output_arr = []
    for batch in range(int(int(len(test_data)) / batch_size)):
       
        test_data_new = test_data[batch*batch_size: (batch+1)*batch_size]
        #test_label_new = test_label[batch*batch_size: (batch+1)*batch_size]
        new_batch_size = len(train_data_new)
        
        for iter in range(len(test_data_new)):
            test_l3 = forward_pass(
                test_data_new[iter], bias3, layer_num=3, weights=weights3, iteration=x+j)
            test_l4 = forward_pass(
                test_l3[0], bias4, layer_num=4, weights=weights4, iteration=x+j)
            output_arr.append(test_l4[0])
    arr = []
    for item in range(len(output_arr)):
        if output_arr[item] <= 0:
            output_arr[item] = int(0)
            arr.append(int(0))
        else:
            output_arr[item] = int(1)
            arr.append(int(1))

    arr = np.array(arr)
    accuracy2(test_label, arr, epoch=0)
    arr.tofile('test_predictions.csv', sep='\n')
    print("Time of testing", (time.time()) - start)