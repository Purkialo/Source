import numpy as np
import pandas as pd
import pylab
import math
"""
Created on Wen Oct 3 2018
@author: Purkialo
"""

"""
相关性分析与线性回归
线性回归未完成
"""
def loaddata():
    data_xls = pd.read_excel('2014 and 2015 CSM dataset.xlsx', index_col=0)
    data_xls.to_csv('dataset.csv', encoding='utf-8')
    data = pd.read_csv('dataset.csv', sep=',')
    delnum = [0,1,6,-1]
    data.drop(data.columns[delnum],axis=1,inplace=True)
    corr = data.corr().values
    mydel = []
    for i in range(len(corr[0])):
        if(corr[i,2] <= 0.2 and corr[i,2] >= -0.2):
            mydel.append(i)
    data.drop(data.columns[mydel],axis=1,inplace=True)
    data = np.array(data.values)
    data[:, [1, -1]] = data[:, [-1, 1]]
    return data

def compute_error(b,a1,a2,a3,data):

    totalError = 0.00
    x1 = data[:,0]
    x2 = data[:,1]
    x3 = data[:,2]
    y = data[:,-1]
    for i in range(len(data)):
        x1 = data[i,0]
        x2 = data[i,1]
        x3 = data[i,2]
        y = data[i,-1]
        buffer = (y-(a1*x1+a2*x2+a3*x3+b)**2)/10000000000000
        if not(math.isnan(buffer)):
            totalError += buffer
    print(totalError)
    return totalError/float(len(data))

def optimizer(data,starting_b,starting_a1,starting_a2,starting_a3,learning_rate,num_iter):
    b = starting_b
    a1 = starting_a1
    a2 = starting_a2
    a3 = starting_a3

    #梯度下降
    for i in range(5):
        #更新参数
        b,a1,a2,a3 =compute_gradient(b,a1,a2,a3,data,learning_rate)
        if i%100==0:
            print ('iter {0}:error={1}'.format(i,compute_error(b,a1,a2,a3,data)))
    return [b,a1,a2,a3]

def compute_gradient(b_current,a1_current,a2_current,a3_current,data,learning_rate):

    b_gradient = 0
    a1_gradient = 0
    a2_gradient = 0
    a3_gradient = 0

    N = float(len(data))
    #Two ways to implement this
    #first way
    # for i in range(0,len(data)):
    #     x = data[i,0]
    #     y = data[i,1]
    #
    #     #computing partial derivations of our error function
    #     #b_gradient = -(2/N)*sum((y-(m*x+b))^2)
    #     #m_gradient = -(2/N)*sum(x*(y-(m*x+b))^2)
    #     b_gradient += -(2/N)*(y-((m_current*x)+b_current))
    #     m_gradient += -(2/N) * x * (y-((m_current*x)+b_current))

    #Vectorization implementation
    x1 = data[:,0]
    x2 = data[:,1]
    x3 = data[:,2]
    y = data[:,-1]
    b_gradient = -(2/N)*(y-a1_current*x1-a2_current*x2-a3_current*x3-b_current)
    b_gradient = np.sum(b_gradient,axis=0)
    a1_gradient = -(2/N)*x1*(y-a1_current*x1-a2_current*x2-a3_current*x3-b_current)
    print(a1_gradient[0])
    a1_gradient = np.sum(a1_gradient,axis=0)
    a2_gradient = -(2/N)*x2*(y-a1_current*x1-a2_current*x2-a3_current*x3-b_current)
    a2_gradient = np.sum(a2_gradient,axis=0)
    a3_gradient = -(2/N)*x3*(y-a1_current*x1-a2_current*x2-a3_current*x3-b_current)
    a3_gradient = np.sum(a3_gradient,axis=0)
    #update our b and m values using out partial derivations
    new_b = b_current - (learning_rate * b_gradient)
    new_a1 = a1_current - (learning_rate * a1_gradient)
    new_a2 = a2_current - (learning_rate * a2_gradient)
    new_a3 = a3_current - (learning_rate * a3_gradient)
    return [new_b,new_a1,new_a2,new_a3]

def Linear_regression(data):
    #建模  y = a1x1 + a2x2 + a3x3 +b
    learning_rate = 0.001 #学习率
    initial_b = 1
    initial_a1 = 100
    initial_a2 = 1000
    initial_a3 = 0.001
    num_iter = 1000 #训练迭代次数

    #训练
    print ('initial variables:\n initial_b = {0}\n intial_a1 = {1}\n intial_a2 = {2}\n intial_a3 = {3}\n error of begin = {4} \n'\
        .format(initial_b,initial_a1,initial_a2,initial_a3,compute_error(initial_b,initial_a1,initial_a2,initial_a3,data)))

    #梯度下降优化
    [b,a1,a2,a3] = optimizer(data,initial_b,initial_a1,initial_a2,initial_a3,learning_rate,num_iter)

    print ('final formula parmaters:\n b = {0}\n a1 = {1}\n a2 = {2}\n a3 = {3}\n error of end = {4} \n'\
        .format(b,a1,a2,a3,compute_error(b,a1,a2,a3,data)))

if __name__ =='__main__':
    data = loaddata()
    Linear_regression(data)
    