# -*- coding: utf-8 -*-
"""
Created on Wen Oct 3 2018
@author: Purkialo
"""
import numpy as np

K_num = 20

def calcu(input,dataset):
    dist = []
    for i in range(length_train):
        dist.append(np.linalg.norm(input - dataset[i]))
    dist = np.array(dist,dtype = float).reshape(length_train,1)
    dist = np.hstack((dataname.reshape(length,1)[:length_train],dist))
    dist = np.array(sorted(dist,key = lambda result:float(result[1])))
    return dist

def count(input):
    key = np.unique(input[:,0])
    result = {}
    for k in key:
        mask = (input == k)
        arr_new = input[mask]
        v = arr_new.size
        result[k] = v
    result = sorted(result.items(),key=lambda result:result[1],reverse=True) 
    return result

data = []
name = []

with open("letter-recognition.data") as file:
    for line in file:
        data.append(line.strip().split(',')[1:])
        name.append(line.strip().split(',')[0])
length = len(name)
dataset = np.array(data,dtype=int)
dataname = np.array(name)
length_train = int(0.95 * length)
length_test = length - length_train
testset = dataset[length_train:]
testset = dataset[length_train:].reshape(length_test,16)
testname = dataname[length_train:].reshape(length_test,1)
flag_r = 0
flag_w = 0
for i in range(length_test):
    result = calcu(testset[i],dataset)
    dic = count(result[0:K_num])
    if(dic[0][0] != testname[i][0]):
        print(dic[:3])
        flag_w = flag_w + 1
        print("Predicted letter: %s, actually: %s, it's wrong!" % (str(dic[0][0]),str(testname[i][0])))
    else:
        flag_r = flag_r + 1
        print("Predicted letter: %s, actually: %s, it's right!" % (str(dic[0][0]),str(testname[i][0])))
print("Accuracy: ",flag_r/(flag_r + flag_w))