# -*- coding: utf-8 -*-
"""
Created on Wen Oct 3 2018
@author: Purkialo
"""

"""
贝叶斯分类
"""
import numpy as np

def loaddata():
    data = []
    name = []
    buffer = []
    with open("nursery.data") as file:
        for line in file:
            buffer = line.strip().split(',')
            buffer = [i for i in buffer if i != '']
            if(len(buffer)):
                data.append(buffer)
    length = len(data)
    length_train = int(0.95 * length)
    return np.array(data[:length_train]),np.array(data[length_train:])

def findvalue(input):
    res = []
    for i in input:
        if not(i in res):
            res.append(i)
    return res

def arrangedata(data_train,res_num,res_lable):
    data_train = np.array(sorted(data_train,key = lambda result:result[-1]))
    data_train_buffer = [i for i in data_train if(i[-1] == res_lable)]
    res = {}
    for i in range(len(data_train[0]) - 1):
        if(i not in res.keys()):
            res[i] = {}
        atta_lable = findvalue(data_train[:,i])
        for j in range(len(atta_lable)):
            res[i][atta_lable[j]] = float(len([k for k in data_train_buffer if(k[i] == atta_lable[j])])) / res_num
    return res

def bias(prob_map,test_data,res_prob,res_lable):
    res = []
    for i in range(len(res_prob)):
        res.append(res_prob[i])
        for j in range(len(test_data)):
            res[i] *= prob_map[res_lable[i]][j][test_data[j]]
    return res

if __name__ == '__main__':
    data_train,data_test = loaddata()
    length_train = len(data_train)
    length_label = len(data_train[0]) - 1
    lable_value = []
    for i in range(length_label): 
        lable_value.append(findvalue(data_train[:,i]))
    res_lable = findvalue(data_train[:,-1])     #所有结果的标签
    res_num = []                                #每类结果的个数
    lable_train = data_train[:,-1]
    prob_dic = {} #p(属性|类别)
    for i in range(len(res_lable)):
        res_num.append(np.sum(lable_train == res_lable[i]))
        prob_dic[res_lable[i]] = arrangedata(data_train,res_num[i],res_lable[i])  
    res_prob = [i/length_train for i in res_num]#每个大类结果的概率
    flag_w = 0    
    for i in range(len(data_test)):
        my_pre_prob = bias(prob_dic,data_test[i][:-1],res_prob,res_lable)   #预测概率
        my_pre_lable = res_lable[my_pre_prob.index(max(my_pre_prob))]        #预测结果
        true_lable = data_test[i][-1]
        if(true_lable != my_pre_lable):
            flag_w = flag_w + 1
            print("Predicted class: %s, actually: %s, it's wrong!" % (str(my_pre_lable),str(true_lable)))
            checkwindow = {}
            for j in range(len(res_lable)):
                checkwindow[res_lable[j]] = my_pre_prob[j]
            print(checkwindow)
        else:
            print("Predicted class: %s, actually: %s, it's right!" % (str(my_pre_lable),str(true_lable)))
    
    print("Accuracy: ",(len(data_test) - flag_w)/len(data_test))    
    #for i in range(len(res_lable)):
    #    print(prob_dic[res_lable[i]])