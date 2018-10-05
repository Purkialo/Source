# -*- coding: utf-8 -*-
"""
Created on Wen Oct 3 2018
@author: Purkialo
"""
import numpy as np
import math

def loaddata():
    data = []
    name = []
    buffer = []
    with open("page-blocks.data") as file:
        for line in file:
            buffer = line.strip().split(' ')
            buffer = [i for i in buffer if i != '']
            data.append(buffer[:-1])
            name.append(buffer[-1])
    length = len(name)
    length_train = 100#int(0.95 * length)
    return np.array(data[:length_train]),np.array(data[length_train:]),np.array(name[:length_train]),np.array(name[length_train:])

def calu_D(my_labelset):
    labelset = my_labelset.copy()
    label = {}
    info_D = 0.0
    length_train = len(labelset)
    for i in range(length_train):
        if(labelset[i] in label.keys()):
            label[labelset[i]] += 1
        else:
            label[labelset[i]] = 1
    for labelset[i] in label.keys():
        p = float(label[labelset[i]])/length_train
        info_D -= p * math.log(p,2)
    return info_D
#labelset1为最终label，labelset2是属性分裂,return IGR,返回该属性的IGR
def calu_label_D(input_labelset1,input_labelset2):
    labelset1 = input_labelset1.copy()
    labelset2 = input_labelset2.copy()
    label_num = {} #该属性取值的分布
    label_result = {} #该属性取值对应结果的分布
    total_D = calu_D(labelset1)
    length_train = len(labelset1)
    for i in range(length_train):
        if(labelset2[i] in label_num.keys()):
            label_num[labelset2[i]] += 1
        else:
            label_num[labelset2[i]] = 1
    for i in range(length_train):
        if(labelset2[i] in label_result.keys()):
            if(labelset1[i] in label_result[labelset2[i]].keys()):
                label_result[labelset2[i]][labelset1[i]] += 1
            else:
                label_result[labelset2[i]][labelset1[i]] = 1
        else:
            label_result[labelset2[i]] = {labelset1[i]:1}
    #print(label_num)
    #print(label_result)
    info_D = 0.0
    for i in label_num.keys():
        p1 = float(label_num[i])/length_train
        info_D_buf = 0.0
        for j in label_result[i].keys():
            p2 = float(label_result[i][j])/label_num[i]
            info_D_buf -= p2 * math.log(p2,2)
        info_D += p1 * info_D_buf
    Gain = total_D - info_D
    H = calu_D(labelset2)
    if(H != 0):
        IGR = Gain/H
    else:
        IGR = -1
    return IGR
#修改列表并返回当前情况IGR return res
def caulcu_divideres(mylabelset,divplace,length_train):
    labelset = mylabelset.copy()
    for i in range(length_train):
        if(float(labelset[i][1]) <= float(divplace)):
            labelset[i][1] = 0
        else:
            labelset[i][1] = 1
    res = calu_label_D(labelset[:,0],labelset[:,1])
    return res
#labelset[0]为最终label，labelset[1]是属性分裂,return pos,res返回该属性的分裂点和IGR
def find_divide(labelset,length_train):
    divplace = []
    divresult = []
    labelset = np.array(sorted(labelset,key = lambda result:float(result[1])))
    for i in range(length_train):
        if(labelset[i][1] not in divplace):
            divplace.append(labelset[i][1])
    for i in range(len(divplace) - 1):
        divresult.append(caulcu_divideres(labelset,float(divplace[i]),length_train))
    pos = divplace[divresult.index(max(divresult))]
    res = max(divresult)

    return pos,res
#返回该标签处理后的剩余数据
def arrangedata(data_train,num):
    data_train = np.array(sorted(data_train,key = lambda result:float(result[num])))
    for i in range(len(data_train)):
        if(int(data_train[i][num]) != 0):
            break
    return data_train[:i],data_train[i:]
#生成决策树，使用字典保存
def most_result(label_list):
    label_nums = {} 
    for label in label_list: 
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    label_nums = sorted(label_nums.items(),key=lambda result:result[1],reverse=True)
    return label_nums[0][0]
def tree_generator(IGR,data_train):
    print(IGR)
    for i in range(len(IGR)):
        if(IGR[i] != -1):
            IGR[i] = calu_label_D(data_train[:,-1],data_train[:,i])
    print(IGR)
    label_num = IGR.index(max(IGR))
    label_list = data_train[:,-1]
    if np.sum(label_list == label_list[0]) == len(label_list):
        return label_list[0]
    if IGR.count(-1) == len(IGR):
        return most_result(label_list)
    it_decision_tree = label_num
    IGR[label_num] = -1
    decision_tree = {it_decision_tree:{'pos':{},0:{},1:{}}}
    decision_tree[it_decision_tree]['pos'] = div_point[label_num]
    """data_train0,data_train1 = arrangedata(data_train,9)
    print(IGR)
    IGR0 = []
    IGR1 = []
    for i in range(len(IGR)):
        if(IGR[i] != -1):
            IGR0[i] = calu_label_D(data_train0,data_train0[:,-1])
            IGR1[i] = calu_label_D(data_train1,data_train1[:,-1])
        else:
            IGR0[i] = -1
            IGR1[i] = -1
    decision_tree[it_decision_tree][0] = tree_generator(IGR0,data_train0)
    decision_tree[it_decision_tree][1] = tree_generator(IGR1,data_train1)"""
    return decision_tree

if __name__ == '__main__':
    data_train,data_test,label_train,label_test = loaddata()
    length_train = len(label_train)
    length_label = len(data_train[0])
    div_point = []
    div_IGR = []
    for i in range(length_label):
        label = np.hstack((label_train.reshape(length_train,1),data_train[:,i].reshape(length_train,1)))
        pos,res = find_divide(label,length_train)
        div_point.append(pos)
        div_IGR.append(res)
    for i in range(length_train):
        for j in range(length_label):
            if(float(data_train[i][j]) <= float(div_point[j])):
                data_train[i][j] = 0
            else:
                data_train[i][j] = 1
    data_train = np.hstack((data_train,label_train.reshape(length_train,1)))
    #print(data_train)#[:,[1,-1]])
    #已取得修改后的离散性数据data_train
    decision_tree = tree_generator(div_IGR,data_train)
    print(decision_tree)