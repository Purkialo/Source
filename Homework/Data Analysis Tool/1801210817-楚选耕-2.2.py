import numpy as py

def loaddata():
    data = []
    name = []
    with open("page-blocks.data") as file:
        for line in file:
            data.append(line.strip().split(',')[:-1])
            name.append(line.strip().split(',')[-1])
    length = len(name)
    length_train = int(0.95 * length)
    length_test = length - length_train
