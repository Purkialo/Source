# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import math
"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""
def Gradient_Ascent_test():
	def f_prime(x_old):									#f(x)的导数
		return -2 * x_old + 4
	x_old = -1											#初始值，给一个小于x_new的值
	x_new = 0											#梯度上升算法初始值，即从(0,0)开始
	alpha = 0.01										#步长，也就是学习速率，控制更新的幅度
	presision = 0.00000001								#精度，也就是更新阈值
	while abs(x_new - x_old) > presision:
		x_old = x_new
		x_new = x_old + alpha * f_prime(x_old)			#上面提到的公式
	print(x_new)										#打印最终求解的极值近似值

"""
函数说明:加载数据

Parameters:
	无
Returns:
	dataMat - 数据列表
	labelMat - 标签列表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""
def loadDataSet():
	dataMat = []														#创建数据列表
	labelMat = []														#创建标签列表
	fr = open('testSet.txt')											#打开文件	
	for line in fr.readlines():											#逐行读取
		lineArr = line.strip().split()									#去回车，放入列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#添加数据
		labelMat.append(int(lineArr[2]))								#添加标签
	fr.close()															#关闭文件
	return dataMat, labelMat											#返回
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
    datanum = []
    for i in range(len(data)):
    	buffer = [i]
    	datanum.append(buffer)
    data[:, [1, -1]] = data[:, [-1, 1]]
    data = np.concatenate((datanum,data),axis = 1)
    return data
"""
函数说明:sigmoid函数

Parameters:
	inX - 数据
Returns:
	sigmoid函数
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法

Parameters:
	dataMatIn - 数据集
	classLabels - 数据标签
Returns:
	weights.getA() - 求得的权重数组(最优参数)
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""
def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
	labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
	m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
	alpha = 0.001														#移动步长,也就是学习速率,控制更新的幅度。
	maxCycles = 500														#最大迭代次数
	print(m,n)
	weights = np.ones((n,1))
	print(weights)
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()												#将矩阵转换为数组，返回权重数组

def judge(dataMatIn,weights):
	res = 0
	for i in range(len(weights) - 1):
		res += dataMatIn[i] * weights[i]
	res += weights[-1]
	if(res >= 0):
		return 1
	else:
		return 0
	return res
if __name__ == '__main__':
	#dataMat, labelMat = loadDataSet()
	data = loaddata()
	length_data = len(data)
	train_data = int(length_data * 0.8)
	print(len)
	for i in data:
		if(i[-1] < 1e06):
			i[-1] = 0
		else:
			i[-1] = 1
	dataMat = data[:train_data,:-1]
	labelMat = data[:train_data,-1]
	weights = gradAscent(dataMat, labelMat)
	print(weights)
	flag_r = 0
	for i in range(length_data - train_data):
		res = judge(data[train_data + i,:-1],weights)
		if(res != data[train_data + i,-1]):
			print("Predicted wrong!")
		else:
			flag_r += 1
			print("Predicted right!")
	print("Accuracy: ",flag_r/(length_data - train_data))