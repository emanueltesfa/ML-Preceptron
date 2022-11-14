import numpy as np
import sys
import argparse as parser
import csv

def parse_inputs(csv2): 
    with open(f"public/{csv2}",'r') as file:
        my_reader = csv.reader(file, delimiter=',')
    return my_reader
        

if __name__ == "__main__":
    # python3 NeuralNetwork.py circle_train_data.csv circle_train_label.csv circle_test_data.csv                


    counter = 0
    for i in sys.argv:
        if counter == 1 :  train_data = parse_inputs(i)
        elif counter == 2 : train_label = parse_inputs(i)
        elif counter == 3 :  test_data = parse_inputs(i)
        elif counter == 4: test_label = parse_inputs(i)
        counter += 1
    print(train_data)
    

    
    