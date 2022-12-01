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


# bias = 0 #value of bias
epochs = 50
batch_size = 20
neurons = 8
input_size = 2
learning_rate = 0.03


def parse_inputs(csv):
    str = f'public/{csv}'
    return genfromtxt(str, delimiter=',')


# layer by layer oeprations
# number of times calculator is dependent upon how many layers i have
# size of wieght matrix changes based how many neurons in a layer
def d_activation_tanh(input_val, prev_deriv):
    return (1 - (np.tanh(input_val)**2)) * prev_deriv


def activation_tanh(input_val):
    return np.tanh(input_val)


def activation_sigmoid(input_val):
    # return np.softmaz()
    # for item in input_val:
    val = 1/(1 + np.exp(-1*input_val))
    # print(" val :", val)

    return val


# x is input before
# y
# fdor last layer is deriv loss function (matrix)
# for all other layers deriv = input @ prev deriv #        backprop(layer4_data[1], 0, layer4_data[0], d_loss_mat )

def d_activation_sigmoid(prev_deriv, x):
    val = activation_sigmoid(x) * (1 - activation_sigmoid(x))
    # print("VAL SHPE" , prev_deriv.shape)
    val *= prev_deriv
    return val


# calc all wieghts at once and feed back
def backprop(weights, input, der_loss, prev_bias, layer_num):
    # m = input.shape[1]
    input = input.T
    # print("\n\weights backprop: ", weights.shape, "der loss: ",
    # der_loss.shape, "input: ", input.shape)
    # derloss instead if not wokring
    dw = np.dot(input.reshape(-1, 1), der_loss.reshape(1, -1))
    # print( "dw,", dw)
    db = der_loss
    # prev_bias = prev_bias - (db * learning_rate)
    # print("db", db.shape)
    # weights = weights - ( dw * learning_rate )
    return [der_loss @ weights.T, dw, db]  # test shape

# t_bias3, bias3, t_bias4, bias4, t_weights3, weights3, t_weights4, weights4, new_batch_size


def set_weights_bias(t_b3, b3, t_b4, b4, t_w3, w3, t_w4,  w4, new_batch_size):
    # print("ENTER SET WEIGHTS")
    # time.sleep(1)
    # print(b3.shape, t_b3.shape)
    b3 = b3 - (t_b3 * learning_rate) / new_batch_size
    b4 -= (t_b4 * learning_rate) / new_batch_size
    w3 -= (t_w3 * learning_rate) / new_batch_size
    w4 -= (t_w4 * learning_rate) / new_batch_size
    return b3, b4, w3, w4


def leaky_Relu(x):
    new_arr = copy.deepcopy(x)
    # print("leaky")
    """for ix, iy in np.ndindex(x.shape):
        # print(x[])
        if x[ix, iy] < 0:
            x[ix, iy] *= 0.01
    return x"""
    new_arr = new_arr.reshape(-1, 1)
    # print(x.shape)
    for i in range(len(new_arr)):
        if new_arr[i] < 0:
            new_arr[i] *= 0.01
    return new_arr.T
    # return np.where(x > 0, x, x * 0.01)


# def d_leaky_Relu(x):
    # print("leaky")
    """for ix, iy, in np.ndindex(x.shape):
        print()
        if x[ix, iy] < 0:
            x[ix, iy] = 0.01
        else:
            x[ix, iy] = 1
    return x"""


def d_leaky_Relu(old_x, alpha=0.01):
    x = copy.deepcopy(old_x)
    x = x.reshape(-1, 1)
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.01
        else:
            x[i] = 1
    return x.T
# return np.where(x>0, 1, alpha)


def accuracy(label, pred, epoch):

    counter = 0
    pred = np.array(pred)
    #print("acc label: ", np.array(label[:20]))
    #print("prediction: ", np.array(pred[:20]))
    # pred = np.round(pred)

    for i in range(len(pred)):
        if pred[i] >= 0:
            pred[i] = 1.
        else:
            pred[i] = -1.
        # print(pred[i], label[i])
        if label[i] == pred[i]:
            counter += 1
    print("Num correct predictions is: ", counter)


def loss(label, output):  # cost is matrix
    # print("label ", label, "OUTPUT  ", output)
    cost = np.sum(label * np.log(output) + (1-label) * np.log(1 - output))

    return cost


def back_loss(label, output):  # d_csorss * w.T
    label = label.reshape(1, 1)
    # print("label.shape: ", label.shape, "output:", output.shape )

    d_cross = ((label/output) - (1-label)/(1 - output))
    # print(d_cross.shape)
    return d_cross


def d_mean_squared_loss(label, output_val):
    return (output_val - label)


# output network - excpted
def forward_pass(input1, bias, layer_num,  weights, iteration):
    # batch one size 3
    output = None
    # print("Weights: ",np.shape( weights), "layer num: ", layer_num )

    # print("bias is: ", bias)
    output_temp = np.dot(input1, weights) + bias  # np.matmul(input1, weights )
    # print("output aftre dot, ", output_temp)
    if layer_num == 4:
        output = activation_tanh(output_temp)

    else:
        output = activation_tanh(output_temp)
    # print("weights in forward, ", weights)
    # print("output after relu, ", output)

    return [output, weights, output_temp, input1]


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

    weights1, weights2, weights3, weights4, counter = 0, 0, 0, 0, 0
    bias3, bias4 = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(1, 8), 0.1

    # print("bias3", bias3.shape)
    t_weights3, t_weights4 = np.zeros((2, 8)), np.zeros((8, 1))
    # print(t_weights3)
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
    # print(train_label)

    weights4 = np.random.normal(size=(neurons, 1))
    # print("WEIGHTS UPDATED")

    weights3 = np.random.normal(size=(input_size, neurons))
    # print("WEIGHTS UPDATED")
    train_label = tanh_fix(train_label)
    # print("train_label_final1: ", train_label_final)
    for x in range(epochs):
        # print(int((len(train_data))/ batch_size))
        total_loss = 0
        final_output_mat = []
        final_label_mat = []
        # time.sleep(1)
        # print("\n\nepoch is: ", x)

        for j in range(int(int(len(train_data)) / batch_size)):
            # print("Epoch is is: ", x)
            # print("train_label_final2 : ", train_label_final)

            train_data_new = train_data[j*batch_size: (j+1)*batch_size]
            train_label_new = train_label[j*batch_size: (j+1)*batch_size]
            new_batch_size = len(train_data_new)
            # print("train_label_final3: ", train_label_final)

            # print("train label old", train_label_new)

            # train_label_new = tanh_fix(train_label_new)
            # print("train_label_final4: ", train_label_final)

            # print(train_label_new)
            # time.sleep(1)
            for i in range(len(train_data_new)):  # iterate items in batch

                # parse datum for minibatch sizes

                # print("train label new", train_label_new)

                # train_data_new = train_data[0]
                # train_label_new = train_label[0]
                # print(train_label[0])
                # print("Input data: ", train_data_new)

                """if j == 0 and x == 0:
                     layer1_data = forward_pass(
                         train_data_new, layer_num=1, weights=None, iteration=x+j)
                weights1 = layer1_data[1

                    layer2_data = forward_pass(
                        layer1_data[0], layer_num=2, weights=None, iteration=x+j)
                weights2 = layer2_data[1

                    layer3_data = forward_pass(
                        train_data_new[i], bias3, layer_num=3, weights=None, iteration=x+j)
                    weights3 = layer3_data[1]

                    layer4_data = forward_pass(
                        layer3_data[0], bias4, layer_num=4, weights=None, iteration=x+j)
                    weights4 = layer4_data[1]"""

                # else:  # use weights from first iteration
                """layer1_data = forward_pass(train_data_new, layer_num = 1, weights = weights1, iteration = x+j)
                weights1 = layer1_data[1]"""
                """layer2_data = forward_pass(layer1_data[0], layer_num = 2, weights = weights2, iteration = x+j)
                weights2 = layer2_data[1]"""
                layer3_data = forward_pass(
                    train_data_new[i], bias3, layer_num=3, weights=weights3, iteration=x+j)
                # t_weights3 += layer3_data[1]

                layer4_data = forward_pass(
                    layer3_data[0], bias4, layer_num=4, weights=weights4, iteration=x+j)
                # t_weights4 = layer4_data[1]

                # print("layer4_data[0][0]", layer4_data[0][0])
                # print(lay)
                final_output = layer4_data[0][0]
                # print("output", layer4_data[0][0])
                # print("exptected: ", train_label_new[i])
                # print("final output[0]: ", final_output[0])
                final_output_mat.append(final_output[0])
                # print("train_label_final5: ", train_label_final)
                final_label_mat.append(train_label_new[i])

                # total_loss += mean_squared_loss(train_label_new, final_output)
                # accuracy(train_label_new, final_output_mat ,epoch = x)
                # print(j)
                # print("loss calc: ", total_loss)
                """if j == 99:
                    # print(np.array(final_output_mat).shape)

                    print("epoch", x)"""
                # print(total_loss)
                # print(np.array(final_label_mat).shape)
                # print("output, ", final_output_mat[:100])
                # print("label", final_label_mat[:100])
                # accuracy(final_label_mat, final_output_mat, epoch=x)
                # print("total_loss: ", total_loss)
                # if j != 0:

                # d_loss_mat = back_loss(train_label_new, layer4_data[0])

                # !! return of back prop [der_loss @ weights.T, weights, prev_bias ] !!

                # layer1-4_data     return foward feeding [output, weights, output_temp, input1] !!!
                # 0      1          2         3
                # print("train_label_final 6: ", train_label_final)

                d_loss_mat = d_mean_squared_loss(
                    train_label_new[i], layer4_data[0])

                d_tan4 = d_activation_tanh(
                    layer4_data[2], d_loss_mat)  # FUNCTION
                bp_lay4 = backprop(weights4, layer4_data[3], d_tan4, bias4, 4)
                t_weights4 = bp_lay4[1]
                t_bias4 = bp_lay4[2]
                # print("bias4 after update, " ,bias4)
                # print("weights4 : ", weights4)
                # print(temp_w, weights4)
                # assert( np.alltrue(temp_w == weights4) )

                # print("\n\ndtan4", d_tan4, "output layer 4", bp_lay4)
                # print("weights4", weights4)

                # print("dtan3", d_leaky3, "output layer 3", bp_lay3 )
                # print("weights3", weights3)

                # dLossmat => bp_lay4 change
                d_leaky3 = d_activation_tanh(layer3_data[2], bp_lay4[0])
                bp_lay3 = backprop(
                    weights3, layer3_data[3], d_leaky3, bias3, 3)
                t_weights3 = bp_lay3[1]
                # print("temp weight: ", t_weights3)
                t_bias3 = bp_lay3[2]

            output_bias_weights = set_weights_bias(t_bias3, bias3, t_bias4, bias4, t_weights3, weights3,
                                                   t_weights4, weights4, new_batch_size)
            bias3 = output_bias_weights[0]
            bias4 = output_bias_weights[1]
            weights3 = output_bias_weights[2]
            weights4 = output_bias_weights[3]

        """final_output = layer4_data[0][0]
        print("output", layer4_data[0][0])
        # print("exptected: ", train_label_new[i])
        final_output_mat.append(final_output[0])
        final_label_mat.append(train_label_new)"""
        # print("train_label_final 7: ", train_label_final)

        accuracy(train_label, final_output_mat, epoch=x)

        # print("dtan3", d_leaky3, "output layer 3", bp_lay3 )
        # print("weights3", weights3)

        # print(bp_lay3[0].shape)

        """d_tan2 = d_leaky_Relu( layer2_data[2]) *  bp_lay3[0]  # dLossmat => bp_lay3 change
                d_tan2 *= bp_lay3[0]
                bp_lay2 = backprop(layer2_data[1], layer1_data[0], d_tan2, 2)
                weights2 = bp_lay2[1]
                # print(bp_lay2[0].shape)"""

        # d_tan1 = d_leaky_Relu(layer1_data[2]) * bp_lay2[0]  # dLossmat => bp_lay2 change
        # d_tan1 *= bp_lay2[0]
        """bp_lay1 = backprop(layer1_data[1], train_data_new, d_sig,1)
                weights1 = bp_lay1[1]"""
        # print("epoch: ",x, "batch num: " ,j, "After backprop weights", weights4, " output: ", layer4_data[0], "exptected: ", train_label_new)
    # print line during foward pass

    test_l3 = forward_pass(
        test_data, bias3, layer_num=3, weights=weights3, iteration=x+j)
    test_l4 = forward_pass(
        test_l3[0], bias4, layer_num=4, weights=weights4, iteration=x+j)
    # print(test_l4[0])
    arr = []
    for item in range(len(test_l4[0])):
        if test_l4[0][item] <= 0:
            test_l4[0][item] = int(0)
            arr.append(int(0))
        else:
            test_l4[0][item] = int(1)
            arr.append(int(1))

    arr = np.array(arr)
    arr.tofile('data.csv', sep='\n')
    accuracy(test_label, arr, epoch=0)
